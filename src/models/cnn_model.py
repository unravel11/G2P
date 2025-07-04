"""
CNN模型实现
使用PyTorch实现一维CNN用于基因型-表型预测
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
from .base import BaseModel
import os
from torch.utils.data import DataLoader, TensorDataset
import logging
import matplotlib.pyplot as plt
import time

logger = logging.getLogger(__name__)

# 设置PyTorch的并行计算线程数
torch.set_num_threads(os.cpu_count())

class PositionalEncoding(nn.Module):
    """位置编码层"""
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入张量 [batch_size, seq_len, d_model]
        """
        seq_len = x.size(1)
        return x + self.pe[:1, :].to(x.device)

class CNNModel(BaseModel):
    """CNN模型类"""
    
    class CNN(nn.Module):
        def __init__(self, input_size: int, hidden_sizes: List[int] = [64, 32], dropout_rate: float = 0.2):
            """
            初始化CNN模型
            
            Args:
                input_size: 输入特征维度
                hidden_sizes: 隐藏层大小列表
                dropout_rate: Dropout比率
            """
            super().__init__()
            
            # 输入维度调整层
            self.input_adapter = nn.Linear(input_size, hidden_sizes[0])
            self.batch_norm1 = nn.BatchNorm1d(hidden_sizes[0])
            
            # 构建CNN层
            layers = []
            in_channels = 1
            seq_len = hidden_sizes[0]
            
            for hidden_size in hidden_sizes:
                layers.extend([
                    nn.Conv1d(in_channels, hidden_size, kernel_size=3, padding=1),
                    nn.BatchNorm1d(hidden_size),
                    nn.ReLU(),
                    nn.MaxPool1d(2),
                    nn.Dropout(dropout_rate)
                ])
                in_channels = hidden_size
                seq_len = seq_len // 2
            
            self.cnn = nn.Sequential(*layers)
            
            # 计算最终特征维度
            final_size = seq_len * hidden_sizes[-1]
            
            # 全连接层
            self.fc = nn.Sequential(
                nn.Linear(final_size, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(64, 32),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(32, 1)
            )
            
            # 残差连接
            self.residual = nn.Sequential(
                nn.Linear(input_size, 32),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.Linear(32, 1)
            )
        
        def forward(self, x):
            # 保存原始输入用于残差连接
            original_x = x
            
            # 输入维度调整
            x = self.input_adapter(x)
            x = self.batch_norm1(x)
            x = F.relu(x)
            
            # 添加通道维度
            x = x.unsqueeze(1)
            x = self.cnn(x)
            x = x.view(x.size(0), -1)
            
            # CNN特征
            cnn_out = self.fc(x)
            
            # 残差连接
            residual_out = self.residual(original_x)
            
            # 合并输出
            return cnn_out + residual_out
    
    def __init__(self, model_params: Optional[Dict[str, Any]] = None, task_type: str = 'regression'):
        """
        初始化CNN模型
        
        Args:
            model_params: 模型参数字典
            task_type: 任务类型，'classification' 或 'regression'
        """
        super().__init__(model_params)
        self.task_type = task_type
        self.feature_names = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.scheduler = None
        self.scaler = StandardScaler()
        self._init_model()
    
    def _init_model(self) -> None:
        """初始化模型"""
        default_params = {
            'hidden_sizes': [128, 64],  # 增加网络容量
            'dropout_rate': 0.5,        # 增加dropout
            'learning_rate': 0.0001,    # 降低学习率
            'batch_size': 32,           # 减小批次大小
            'epochs': 200,
            'early_stopping_patience': 20,
            'weight_decay': 5e-4        # 增加权重衰减
        }
        
        # 更新默认参数
        params = {**default_params, **self.model_params}
        self.model_params = params
        
        # 设置损失函数
        if self.task_type == 'classification':
            self.criterion = nn.CrossEntropyLoss()
        else:
            self.criterion = nn.SmoothL1Loss()  # 使用Smooth L1 Loss
            
        # 创建模型实例
        self.model = self.CNN(
            input_size=1000,  # 默认输入大小,实际训练时会更新
            hidden_sizes=self.model_params['hidden_sizes'],
            dropout_rate=self.model_params['dropout_rate']
        ).to(self.device)
        
        # 使用AdamW优化器
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.model_params['learning_rate'],
            weight_decay=self.model_params['weight_decay']
        )
        
        # 使用余弦退火学习率调度器
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=10,     # 增加重启周期
            T_mult=2,   # 保持周期增长倍数
            eta_min=1e-6  # 最小学习率
        )
    
    def train(self, X: np.ndarray, y: np.ndarray, X_val: np.ndarray = None, y_val: np.ndarray = None, feature_names: Optional[List[str]] = None, output_dir: Optional[str] = None) -> None:
        """
        训练模型
        Args:
            X: 特征矩阵
            y: 目标变量
            X_val: 验证集特征矩阵
            y_val: 验证集目标变量
            feature_names: 特征名称列表
            output_dir: 输出目录
        """
        try:
            # 设置随机种子
            torch.manual_seed(42)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(42)
                torch.cuda.manual_seed_all(42)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            
            # 动态根据输入特征数（SNP数量）初始化模型
            input_size = X.shape[1]
            self.model = self.CNN(
                input_size=input_size,
                hidden_sizes=self.model_params['hidden_sizes'],
                dropout_rate=self.model_params['dropout_rate']
            ).to(self.device)
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.model_params['learning_rate'],
                weight_decay=self.model_params['weight_decay']
            )
            self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=10,
                T_mult=2,
                eta_min=1e-6
            )

            logger.info("开始训练CNN模型...")
            logger.info(f"使用设备: {self.device}")
            logger.info(f"输入数据形状: X={X.shape}, y={y.shape}")
            
            self.feature_names = feature_names
            
            # 标准化表型数据
            logger.info("标准化表型数据...")
            y = self.scaler.fit_transform(y.reshape(-1, 1)).ravel()
            if y_val is not None:
                y_val = self.scaler.transform(y_val.reshape(-1, 1)).ravel()
            
            # 转换为PyTorch张量
            logger.info("转换为PyTorch张量...")
            X = torch.FloatTensor(X)
            y = torch.FloatTensor(y)
            if X_val is not None and y_val is not None:
                X_val = torch.FloatTensor(X_val)
                y_val = torch.FloatTensor(y_val)
            
            # 创建数据加载器
            logger.info("创建数据加载器...")
            dataset = TensorDataset(X, y)
            # 创建固定的随机数生成器
            g = torch.Generator()
            g.manual_seed(42)
            dataloader = DataLoader(
                dataset,
                batch_size=self.model_params['batch_size'],
                shuffle=True,
                num_workers=0,  # 修改为0，避免多进程问题
                pin_memory=True,
                generator=g  # 使用固定的随机数生成器
            )
            
            logger.info("开始训练循环...")
            self.model.train()
            best_loss = float('inf')
            patience_counter = 0
            train_losses = []
            val_losses = []
            
            for epoch in range(self.model_params['epochs']):
                total_loss = 0
                batch_count = 0
                
                try:
                    for batch_X, batch_y in dataloader:
                        try:
                            batch_count += 1
                            # 只在每个epoch开始时输出一次进度
                            if batch_count == 1:
                                logger.info(f"Epoch {epoch+1}/{self.model_params['epochs']} 开始训练...")
                            
                            # 确保数据在正确的设备上
                            batch_X = batch_X.to(self.device, non_blocking=True)
                            batch_y = batch_y.to(self.device, non_blocking=True)
                            if batch_y.dim() == 1:
                                batch_y = batch_y.unsqueeze(1)
                            
                            # 前向传播
                            self.optimizer.zero_grad()
                            outputs = self.model(batch_X)
                            loss = self.criterion(outputs, batch_y)
                            
                            # 反向传播
                            loss.backward()
                            
                            # 梯度裁剪
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                            
                            # 更新参数
                            self.optimizer.step()
                            total_loss += loss.item()
                            
                            # 清理缓存
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                                
                        except RuntimeError as e:
                            logger.error(f"处理batch时出错: {str(e)}")
                            if "CUDA out of memory" in str(e):
                                logger.error("GPU内存不足，尝试减小batch size")
                                raise
                            continue
                    
                    avg_loss = total_loss / len(dataloader)
                    train_losses.append(avg_loss)
                    
                    # 验证集评估
                    if X_val is not None and y_val is not None:
                        self.model.eval()
                        with torch.no_grad():
                            try:
                                X_val_t = X_val.to(self.device, non_blocking=True)
                                y_val_t = y_val.to(self.device, non_blocking=True)
                                if y_val_t.dim() == 1:
                                    y_val_t = y_val_t.unsqueeze(1)
                                val_outputs = self.model(X_val_t)
                                val_loss = self.criterion(val_outputs, y_val_t).item()
                                val_losses.append(val_loss)
                                
                                # 只在验证损失改善时输出详细信息
                                if val_loss < best_loss:
                                    logger.info(f"Epoch {epoch+1}/{self.model_params['epochs']}, "
                                              f"Train Loss: {avg_loss:.6f}, Val Loss: {val_loss:.6f}, "
                                              f"LR: {self.scheduler.get_last_lr()[0]:.6f}")
                                
                                self.model.train()
                                
                                # 更新学习率
                                self.scheduler.step()
                                
                                # Early stopping
                                if val_loss < best_loss:
                                    best_loss = val_loss
                                    patience_counter = 0
                                    logger.info(f"发现更好的模型，验证损失: {val_loss:.6f}")
                                else:
                                    patience_counter += 1
                                    if patience_counter >= self.model_params['early_stopping_patience']:
                                        logger.info(f"Early stopping at epoch {epoch+1}")
                                        break
                                    
                            except RuntimeError as e:
                                logger.error(f"验证集评估时出错: {str(e)}")
                                continue
                    else:
                        # 只在训练损失改善时输出详细信息
                        if avg_loss < best_loss:
                            logger.info(f"Epoch {epoch+1}/{self.model_params['epochs']}, "
                                      f"Loss: {avg_loss:.6f}, LR: {self.scheduler.get_last_lr()[0]:.6f}")
                        
                        # 更新学习率
                        self.scheduler.step()
                        
                        if avg_loss < best_loss:
                            best_loss = avg_loss
                            patience_counter = 0
                        else:
                            patience_counter += 1
                            if patience_counter >= self.model_params['early_stopping_patience']:
                                logger.info(f"Early stopping at epoch {epoch+1}")
                                break
                            
                except Exception as e:
                    logger.error(f"训练过程中出错: {str(e)}")
                    raise
            
            logger.info("模型训练完成")
            
            # 训练结束后绘制loss曲线
            try:
                # 自动生成唯一文件名
                param_str = f"snps{X.shape[1]}_hidden{'-'.join(map(str, self.model_params.get('hidden_sizes', [])))}_dropout{self.model_params.get('dropout_rate', '')}_lr{self.model_params.get('learning_rate', '')}_bs{self.model_params.get('batch_size', '')}"
                timestamp = time.strftime("%Y%m%d%H%M%S")
                
                if output_dir:
                    # 确保输出目录存在
                    os.makedirs(output_dir, exist_ok=True)
                    loss_curve_path = os.path.join(output_dir, f"loss_curve_{param_str}_{timestamp}.png")
                    
                    plt.figure(figsize=(10, 6))
                    plt.plot(train_losses, label='Train Loss', linewidth=2)
                    if val_losses:
                        plt.plot(val_losses, label='Validation Loss', linewidth=2)
                    plt.xlabel('Epoch', fontsize=12)
                    plt.ylabel('Loss', fontsize=12)
                    plt.title('Training/Validation Loss Curve', fontsize=14)
                    plt.legend(fontsize=12)
                    plt.grid(True)
                    plt.tight_layout()
                    
                    # 保存图片
                    plt.savefig(loss_curve_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    logger.info(f'Loss曲线已保存: {loss_curve_path}')
                else:
                    logger.warning("未提供output_dir，无法保存loss曲线")
            except Exception as e:
                logger.error(f'绘制loss曲线出错: {str(e)}')
                logger.error(f'错误详情: {e.__class__.__name__}')
                import traceback
                logger.error(f'错误堆栈: {traceback.format_exc()}')
            
        except Exception as e:
            logger.error(f"训练过程发生错误: {str(e)}")
            raise
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        使用模型进行预测
        
        Args:
            X: 特征矩阵
            
        Returns:
            np.ndarray: 预测结果
        """
        self.model.eval()
        with torch.no_grad():
            X = torch.FloatTensor(X).to(self.device)
            predictions = self.model(X)
            # 反标准化预测结果
            predictions = self.scaler.inverse_transform(predictions.cpu().numpy())
            return predictions.reshape(-1)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        评估模型性能
        
        Args:
            X: 特征矩阵
            y: 真实标签
            
        Returns:
            Dict[str, float]: 评估指标
        """
        y_pred = self.predict(X)
        # 保证一维float类型
        y = np.asarray(y).reshape(-1).astype(np.float64)
        y_pred = np.asarray(y_pred).reshape(-1).astype(np.float64)
        
        # 检查NaN/inf
        if np.any(np.isnan(y)) or np.any(np.isnan(y_pred)):
            raise ValueError("y或y_pred中包含NaN")
        if np.any(np.isinf(y)) or np.any(np.isinf(y_pred)):
            raise ValueError("y或y_pred中包含inf")
        
        if self.task_type == 'classification':
            metrics = {
                'accuracy': accuracy_score(y, y_pred)
            }
        else:
            metrics = {
                'mse': mean_squared_error(y, y_pred),
                'rmse': np.sqrt(mean_squared_error(y, y_pred)),
                'r2': r2_score(y, y_pred),
                'pearson_r': pearsonr(y, y_pred)[0],
                'pearson_p': pearsonr(y, y_pred)[1]
            }
        return metrics
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        获取特征重要性
        
        Returns:
            Dict[str, float]: 特征重要性字典
        """
        if self.model is None:
            raise ValueError("模型尚未训练")
        
        # 使用输入适配器的权重作为特征重要性
        weights = self.model.input_adapter.weight.data.cpu().numpy()
        importances = np.mean(np.abs(weights), axis=0)
        
        if self.feature_names is None:
            self.feature_names = [f"feature_{i}" for i in range(len(importances))]
        
        return dict(zip(self.feature_names, importances))
    
    def save(self, path: str) -> None:
        """
        保存模型到文件
        
        Args:
            path: 保存路径
        """
        if self.model is None:
            raise ValueError("模型尚未训练，无法保存")
        
        # 保存模型状态字典
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'model_params': self.model_params,
            'task_type': self.task_type,
            'feature_names': self.feature_names,
            'scaler': self.scaler
        }, path)
    
    def load(self, path: str) -> None:
        """
        从文件加载模型
        
        Args:
            path: 模型文件路径
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"模型文件不存在: {path}")
        
        # 加载模型状态
        checkpoint = torch.load(path)
        
        # 初始化模型
        self.model_params = checkpoint['model_params']
        self.task_type = checkpoint['task_type']
        self.feature_names = checkpoint['feature_names']
        self.scaler = checkpoint['scaler']
        self._init_model()
        
        # 加载模型参数
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict']) 