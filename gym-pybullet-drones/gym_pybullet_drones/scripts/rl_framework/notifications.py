"""
Notification management for training completion and status updates.
"""

import subprocess
import platform
from typing import Optional


class NotificationManager:
    """Handles system notifications for training completion."""
    
    @staticmethod
    def send_system_notification(title: str, message: str, urgent: bool = False) -> bool:
        """Send system notification (supports macOS and cross-device sync)."""
        try:
            if platform.system() == "Darwin":  # macOS
                sound_name = "Glass" if urgent else "Blow"
                
                applescript = f'''
                display notification "{message}" ¬
                    with title "{title}" ¬
                    subtitle "训练状态更新" ¬
                    sound name "{sound_name}"
                '''
                
                subprocess.run(['osascript', '-e', applescript], check=True)
                print("✅ macOS 系统通知发送成功！")
                print("📱 如果设置正确，iPhone 应该也会收到通知")
                return True
            else:
                print("⚠️  当前系统不支持 macOS 通知")
                return False
        except Exception as e:
            print(f"❌ 系统通知发送失败: {e}")
            return False
    
    @classmethod
    def send_training_completion_notification(
        cls, 
        task: str, 
        episodes: int, 
        final_reward: str, 
        target_reward: float, 
        training_time: float,
        enable_obstacles: Optional[bool] = None
    ) -> bool:
        """Send training completion notification."""
        if isinstance(final_reward, str) or final_reward == "未知":
            success = False
        else:
            try:
                success = float(final_reward) >= target_reward
            except (ValueError, TypeError):
                success = False
                
        status_emoji = "🎉" if success else "⚠️"
        title = f"{status_emoji} 无人机训练完成"
        
        obstacle_info = ""
        if task == "unified" and enable_obstacles is not None:
            obstacle_info = f"\\n障碍物: {'启用' if enable_obstacles else '禁用'}"
        
        message = (f"任务: {task.upper()}\\n轮数: {episodes}\\n"
                  f"性能: {final_reward}/{target_reward}\\n"
                  f"用时: {training_time/3600:.1f}h{obstacle_info}")
        
        return cls.send_system_notification(title, message, urgent=success)
