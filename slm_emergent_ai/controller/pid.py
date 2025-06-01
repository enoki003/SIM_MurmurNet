"""
PID Controller - PID制御の実装

λパラメータの自動調整に使用するPID制御の実装。
"""

class PIDController:
    """
    PID制御クラス
    
    Parameters:
    -----------
    Kp: 比例ゲイン
    Ki: 積分ゲイン
    Kd: 微分ゲイン
    setpoint: 目標値
    """
    def __init__(self, Kp: float, Ki: float, Kd: float, setpoint: float):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.setpoint = setpoint
        
        # 状態変数
        self.prev_error = 0.0
        self.integral = 0.0
    
    def compute(self, process_value: float, dt: float) -> float:
        """
        PID制御値を計算
        
        Parameters:
        -----------
        process_value: 現在の値
        dt: 時間間隔
        
        Returns:
        --------
        制御値
        """
        # 誤差を計算
        error = self.setpoint - process_value
        
        # 積分項を更新
        self.integral += error * dt
        
        # 微分項を計算
        derivative = (error - self.prev_error) / dt if dt > 0 else 0
        
        # PID制御値を計算
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        
        # 状態を更新
        self.prev_error = error
        
        return output
    
    def reset(self):
        """状態をリセット"""
        self.prev_error = 0.0
        self.integral = 0.0
    
    def set_tunings(self, Kp: float, Ki: float, Kd: float):
        """
        ゲインを設定
        
        Parameters:
        -----------
        Kp: 比例ゲイン
        Ki: 積分ゲイン
        Kd: 微分ゲイン
        """
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
    
    def set_setpoint(self, setpoint: float):
        """
        目標値を設定
        
        Parameters:
        -----------
        setpoint: 目標値
        """
        self.setpoint = setpoint
