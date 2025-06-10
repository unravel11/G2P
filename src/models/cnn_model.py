"""
CNN模型实现
使用PyTorch实现一维CNN用于基因型-表型预测
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score
from scipy.stats import pearsonr
from .base import BaseModel
import os
from torch.utils.data import DataLoader, TensorDataset

# 设置PyTorch的并行计算线程数
torch.set_num_threads(os.cpu_count())

class PositionalEncoding(nn.Module):
    """位置编码层"""
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入张量 [batch_size, seq_len, d_model]
        """
        return x + self.pe[:x.size(0)]

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
            
            # 位置编码
            self.pos_encoder = PositionalEncoding(hidden_sizes[0])
            
            # 特征嵌入层
            self.embedding = nn.Sequential(
                nn.LayerNorm(hidden_sizes[0]),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            )
            
            # 构建CNN层
            layers = []
            prev_size = hidden_sizes[0]
            
            for hidden_size in hidden_sizes:
                layers.extend([
                    nn.Conv1d(1, hidden_size, kernel_size=3, padding=1),
                    nn.BatchNorm1d(hidden_size),
                    nn.ReLU(),
                    nn.MaxPool1d(2),
                    nn.Dropout(dropout_rate)
                ])
                prev_size = prev_size // 2
            
            self.cnn = nn.Sequential(*layers)
            
            # 计算最终特征维度
            final_size = prev_size * hidden_sizes[-1]
            
            # 全连接层
            self.fc = nn.Sequential(
                nn.Linear(final_size, 64),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(32, 1)
            )
            
            # 残差连接
            self.residual = nn.Linear(input_size, 1)
        
        def forward(self, x):
            # 保存原始输入用于残差连接
            original_x = x
            
            # 输入维度调整
            x = self.input_adapter(x)
            
            # 位置编码
            x = self.pos_encoder(x)
            
            # 特征嵌入
            x = self.embedding(x)
            
            # 添加通道维度
            x = x.unsqueeze(1)
            x = self.cnn(x)
            x = x.view(x.size(0), -1)
            
            # CNN特征
            cnn_out = self.fc(x)
            
            # 残差连接 - 使用原始输入
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
        self._init_model()
    
    def _init_model(self) -> None:
        """初始化模型"""
        default_params = {
            'hidden_sizes': [128, 64, 32],
            'dropout_rate': 0.3,
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 200,
            'early_stopping_patience': 20,
            'weight_decay': 1e-5
        }
        
        # 更新默认参数
        params = {**default_params, **self.model_params}
        self.model_params = params
        
        # 设置损失函数
        if self.task_type == 'classification':
            self.criterion = nn.CrossEntropyLoss()
        else:
            self.criterion = nn.MSELoss()
    
    def train(self, X: np.ndarray, y: np.ndarray, feature_names: Optional[List[str]] = None) -> None:
        """
        训练模型
        
        Args:
            X: 特征矩阵
            y: 目标变量
            feature_names: 特征名称列表
        """
        self.feature_names = feature_names
        
        # 转换为PyTorch张量
        X = torch.FloatTensor(X)
        y = torch.FloatTensor(y)
        
        # 创建数据集和数据加载器
        dataset = TensorDataset(X, y)
        dataloader = DataLoader(
            dataset,
            batch_size=self.model_params['batch_size'],
            shuffle=True,
            num_workers=os.cpu_count(),  # 使用所有CPU核心
            pin_memory=True  # 如果使用GPU，可以加速数据传输
        )
        
        # 初始化模型
        if self.model is None:
            self.model = self.CNN(
                input_size=X.shape[1],
                hidden_sizes=self.model_params['hidden_sizes'],
                dropout_rate=self.model_params['dropout_rate']
            ).to(self.device)
            
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.model_params['learning_rate'],
                weight_decay=self.model_params['weight_decay']
            )
            
            # 学习率调度器
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=5,
                verbose=True
            )
        
        # 训练模型
        self.model.train()
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.model_params['epochs']):
            total_loss = 0
            for batch_X, batch_y in dataloader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                # 前向传播
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                
                # 反向传播
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                total_loss += loss.item()
            
            # 计算平均损失
            avg_loss = total_loss / len(dataloader)
            
            # 更新学习率
            self.scheduler.step(avg_loss)
            
            # 早停
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.model_params['early_stopping_patience']:
                    break
    
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
            return predictions.cpu().numpy()
    
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
        
        if self.task_type == 'classification':
            metrics = {
                'accuracy': accuracy_score(y, y_pred)
            }
        else:
            metrics = {
                'mse': mean_squared_error(y, y_pred),
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
            'model_params': self.model_params,
            'task_type': self.task_type,
            'feature_names': self.feature_names
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
        self._init_model()
        
        # 加载模型参数
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict']) 