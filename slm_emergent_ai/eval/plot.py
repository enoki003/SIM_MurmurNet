"""
プロット生成モジュール - メトリクスの可視化

システムの評価指標をグラフとして可視化するためのモジュール。
matplotlib と seaborn を使用して、エントロピー、VDI、FCR、速度などの時系列データをプロットする。
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Union, Tuple

# スタイル設定
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("viridis")

class MetricsPlotter:
    """
    メトリクスデータのプロット生成クラス
    """
    def __init__(self, output_dir: str = "./plots"):
        """
        MetricsPlotterの初期化
        
        Parameters:
        -----------
        output_dir: プロット保存ディレクトリ
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def plot_time_series(self, data: Dict[str, List[float]], 
                        title: str = "Metrics Over Time",
                        xlabel: str = "Time Step",
                        ylabel: str = "Value",
                        filename: str = "metrics_time_series.png") -> str:
        """
        時系列データをプロット
        
        Parameters:
        -----------
        data: 時系列データ辞書 {"metric_name": [values...], ...}
        title: プロットタイトル
        xlabel: X軸ラベル
        ylabel: Y軸ラベル
        filename: 保存ファイル名
        
        Returns:
        --------
        保存されたファイルのパス
        """
        plt.figure(figsize=(12, 6))
        
        for metric_name, values in data.items():
            plt.plot(values, label=metric_name)
        
        plt.title(title, fontsize=16)
        plt.xlabel(xlabel, fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.legend()
        plt.tight_layout()
        
        output_path = os.path.join(self.output_dir, filename)
        plt.savefig(output_path, dpi=300)
        plt.close()
        
        return output_path
    
    def plot_entropy_vdi(self, entropy_data: List[float], vdi_data: List[float],
                        title: str = "Entropy and VDI Over Time",
                        filename: str = "entropy_vdi.png") -> str:
        """
        エントロピーとVDIの時系列プロット
        
        Parameters:
        -----------
        entropy_data: エントロピーデータ
        vdi_data: VDIデータ
        title: プロットタイトル
        filename: 保存ファイル名
        
        Returns:
        --------
        保存されたファイルのパス
        """
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        # エントロピーのプロット (左Y軸)
        color = 'tab:blue'
        ax1.set_xlabel('Time Step', fontsize=12)
        ax1.set_ylabel('Entropy', color=color, fontsize=12)
        ax1.plot(entropy_data, color=color, label='Entropy')
        ax1.tick_params(axis='y', labelcolor=color)
        
        # VDIのプロット (右Y軸)
        ax2 = ax1.twinx()
        color = 'tab:orange'
        ax2.set_ylabel('VDI', color=color, fontsize=12)
        ax2.plot(vdi_data, color=color, label='VDI')
        ax2.tick_params(axis='y', labelcolor=color)
        
        # タイトルと凡例
        plt.title(title, fontsize=16)
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, filename)
        plt.savefig(output_path, dpi=300)
        plt.close()
        
        return output_path
    
    def plot_lambda_params(self, lambda_data: Dict[str, List[float]],
                          title: str = "λ Parameters Over Time",
                          filename: str = "lambda_params.png") -> str:
        """
        λパラメータの時系列プロット
        
        Parameters:
        -----------
        lambda_data: λパラメータデータ {"λ_a": [values...], "λ_c": [values...], "λ_s": [values...]}
        title: プロットタイトル
        filename: 保存ファイル名
        
        Returns:
        --------
        保存されたファイルのパス
        """
        plt.figure(figsize=(12, 6))
        
        for param_name, values in lambda_data.items():
            plt.plot(values, label=param_name)
        
        plt.title(title, fontsize=16)
        plt.xlabel('Time Step', fontsize=12)
        plt.ylabel('Parameter Value', fontsize=12)
        plt.legend()
        plt.ylim(0, 1)  # λパラメータは0-1の範囲
        plt.tight_layout()
        
        output_path = os.path.join(self.output_dir, filename)
        plt.savefig(output_path, dpi=300)
        plt.close()
        
        return output_path
    
    def plot_fcr_speed(self, fcr_data: List[float], speed_data: List[float],
                      title: str = "FCR and Speed Over Time",
                      filename: str = "fcr_speed.png") -> str:
        """
        FCRと速度の時系列プロット
        
        Parameters:
        -----------
        fcr_data: FCRデータ
        speed_data: 速度データ
        title: プロットタイトル
        filename: 保存ファイル名
        
        Returns:
        --------
        保存されたファイルのパス
        """
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        # FCRのプロット (左Y軸)
        color = 'tab:green'
        ax1.set_xlabel('Time Step', fontsize=12)
        ax1.set_ylabel('FCR', color=color, fontsize=12)
        ax1.plot(fcr_data, color=color, label='FCR')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.set_ylim(0, 1)  # FCRは0-1の範囲
        
        # 速度のプロット (右Y軸)
        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('Speed (tok/s)', color=color, fontsize=12)
        ax2.plot(speed_data, color=color, label='Speed')
        ax2.tick_params(axis='y', labelcolor=color)
        
        # タイトルと凡例
        plt.title(title, fontsize=16)
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, filename)
        plt.savefig(output_path, dpi=300)
        plt.close()
        
        return output_path
    
    def plot_heatmap(self, data: np.ndarray, 
                    row_labels: List[str], col_labels: List[str],
                    title: str = "Correlation Heatmap",
                    filename: str = "correlation_heatmap.png") -> str:
        """
        ヒートマッププロット
        
        Parameters:
        -----------
        data: 2次元データ配列
        row_labels: 行ラベル
        col_labels: 列ラベル
        title: プロットタイトル
        filename: 保存ファイル名
        
        Returns:
        --------
        保存されたファイルのパス
        """
        plt.figure(figsize=(10, 8))
        
        sns.heatmap(data, annot=True, cmap='viridis', 
                   xticklabels=col_labels, yticklabels=row_labels)
        
        plt.title(title, fontsize=16)
        plt.tight_layout()
        
        output_path = os.path.join(self.output_dir, filename)
        plt.savefig(output_path, dpi=300)
        plt.close()
        
        return output_path
    
    def plot_agent_activity(self, agent_data: Dict[int, List[int]],
                           title: str = "Agent Activity Over Time",
                           filename: str = "agent_activity.png") -> str:
        """
        エージェントの活動量プロット
        
        Parameters:
        -----------
        agent_data: エージェントごとの活動データ {agent_id: [message_counts...], ...}
        title: プロットタイトル
        filename: 保存ファイル名
        
        Returns:
        --------
        保存されたファイルのパス
        """
        plt.figure(figsize=(12, 6))
        
        for agent_id, counts in agent_data.items():
            plt.plot(counts, label=f"Agent {agent_id}")
        
        plt.title(title, fontsize=16)
        plt.xlabel('Time Step', fontsize=12)
        plt.ylabel('Message Count', fontsize=12)
        plt.legend()
        plt.tight_layout()
        
        output_path = os.path.join(self.output_dir, filename)
        plt.savefig(output_path, dpi=300)
        plt.close()
        
        return output_path
    
    def plot_summary(self, metrics_history: Dict[str, List[float]],
                    title: str = "Metrics Summary",
                    filename: str = "metrics_summary.png") -> str:
        """
        メトリクスサマリープロット（複数のサブプロット）
        
        Parameters:
        -----------
        metrics_history: メトリクス履歴データ
        title: プロットタイトル
        filename: 保存ファイル名
        
        Returns:
        --------
        保存されたファイルのパス
        """
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(title, fontsize=16)
        
        # エントロピープロット
        if 'entropy' in metrics_history:
            axs[0, 0].plot(metrics_history['entropy'], color='tab:blue')
            axs[0, 0].set_title('Entropy')
            axs[0, 0].set_xlabel('Time Step')
            axs[0, 0].set_ylabel('Value')
        
        # VDIプロット
        if 'vdi' in metrics_history:
            axs[0, 1].plot(metrics_history['vdi'], color='tab:orange')
            axs[0, 1].set_title('VDI')
            axs[0, 1].set_xlabel('Time Step')
            axs[0, 1].set_ylabel('Value')
        
        # FCRプロット
        if 'fcr' in metrics_history:
            axs[1, 0].plot(metrics_history['fcr'], color='tab:green')
            axs[1, 0].set_title('FCR')
            axs[1, 0].set_xlabel('Time Step')
            axs[1, 0].set_ylabel('Value')
            axs[1, 0].set_ylim(0, 1)
        
        # 速度プロット
        if 'speed' in metrics_history:
            axs[1, 1].plot(metrics_history['speed'], color='tab:red')
            axs[1, 1].set_title('Speed (tok/s)')
            axs[1, 1].set_xlabel('Time Step')
            axs[1, 1].set_ylabel('Value')
        
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, filename)
        plt.savefig(output_path, dpi=300)
        plt.close()
        
        return output_path
    
    def plot_all_metrics(self, metrics_history: Dict[str, List[float]]) -> List[str]:
        """
        すべてのメトリクスプロットを生成
        
        Parameters:
        -----------
        metrics_history: メトリクス履歴データ
        
        Returns:
        --------
        生成されたファイルパスのリスト
        """
        output_files = []
        
        # 時系列プロット
        if all(k in metrics_history for k in ['entropy', 'vdi', 'fcr', 'speed']):
            output_files.append(self.plot_time_series(
                {
                    'Entropy': metrics_history['entropy'],
                    'VDI': metrics_history['vdi'],
                    'FCR': metrics_history['fcr'],
                    'Speed': metrics_history['speed']
                },
                title="All Metrics Over Time",
                filename="all_metrics.png"
            ))
        
        # エントロピーとVDIのプロット
        if all(k in metrics_history for k in ['entropy', 'vdi']):
            output_files.append(self.plot_entropy_vdi(
                metrics_history['entropy'],
                metrics_history['vdi']
            ))
        
        # FCRと速度のプロット
        if all(k in metrics_history for k in ['fcr', 'speed']):
            output_files.append(self.plot_fcr_speed(
                metrics_history['fcr'],
                metrics_history['speed']
            ))
        
        # λパラメータのプロット
        lambda_params = {}
        for key in ['λ_a', 'λ_c', 'λ_s']:
            if key in metrics_history:
                lambda_params[key] = metrics_history[key]
        
        if lambda_params:
            output_files.append(self.plot_lambda_params(lambda_params))
        
        # サマリープロット
        output_files.append(self.plot_summary(metrics_history))
        
        return output_files
