#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
北通游戏手柄控制器模块
支持北通游戏手柄的输入读取和命令转换
"""

import pygame
import numpy as np
import time
from typing import Dict, Optional, Tuple


class BeitongJoystickController:
    """北通游戏手柄控制器类"""
    
    def __init__(self, wait_timeout: float = 10.0):
        """
        初始化北通游戏手柄控制器
        
        Args:
            wait_timeout: 等待手柄连接的超时时间(秒)
        """
        pygame.init()
        pygame.joystick.init()
        
        self.joystick = None
        self.is_connected = False
        
        # 等待手柄连接
        start_time = time.time()
        print("等待北通游戏手柄连接...")
        
        while time.time() - start_time < wait_timeout:
            pygame.event.pump()
            
            if pygame.joystick.get_count() > 0:
                self.joystick = pygame.joystick.Joystick(0)
                self.joystick.init()
                self.is_connected = True
                
                print(f"检测到游戏手柄: {self.joystick.get_name()}")
                print(f"轴数量: {self.joystick.get_numaxes()}")
                print(f"按钮数量: {self.joystick.get_numbuttons()}")
                print(f"帽子数量: {self.joystick.get_numhats()}")
                break
                
            time.sleep(0.1)
        
        if not self.is_connected:
            raise RuntimeError(f"在{wait_timeout}秒内未检测到游戏手柄")
        
        # 控制参数配置
        self.max_linear_speed = 1.5   # 最大线速度 (m/s)
        self.max_angular_speed = 2.0  # 最大角速度 (rad/s)
        self.deadzone = 0.1          # 摇杆死区
        
        # 控制状态 (简化版本，直接启用)
        self.is_locked = False       # 不使用锁定功能
        self.motor_state = 1         # 直接启用电机
        
        # 按钮状态记录 (用于检测按下事件)
        self.button_states = {}
        self.hat_states = {}
        
        print("北通游戏手柄控制器初始化完成!")
        print("控制说明:")
        print("  左摇杆: 前后左右移动")
        print("  右摇杆X轴: 左右旋转")
        print("  (已简化：无锁定功能，直接控制)")
    
    def _apply_deadzone(self, value: float) -> float:
        """应用摇杆死区"""
        if abs(value) < self.deadzone:
            return 0.0
        # 线性映射死区外的值到 [-1, 1]
        if value > 0:
            return (value - self.deadzone) / (1.0 - self.deadzone)
        else:
            return (value + self.deadzone) / (1.0 - self.deadzone)
    
    def _get_button_pressed(self, button_id: int) -> bool:
        """检测按钮是否刚被按下(边沿触发)"""
        current_state = self.joystick.get_button(button_id)
        previous_state = self.button_states.get(button_id, False)
        self.button_states[button_id] = current_state
        return current_state and not previous_state
    
    def _get_hat_pressed(self, hat_id: int, direction: Tuple[int, int]) -> bool:
        """检测帽子键是否刚被按下(边沿触发)"""
        current_state = self.joystick.get_hat(hat_id)
        previous_state = self.hat_states.get(hat_id, (0, 0))
        self.hat_states[hat_id] = current_state
        return current_state == direction and previous_state != direction
    
    def update(self):
        """更新游戏手柄状态"""
        if not self.is_connected:
            return
        
        pygame.event.pump()
        
        # 检查手柄是否仍然连接
        if not self.joystick.get_init():
            self.is_connected = False
            print("游戏手柄连接丢失!")
            return
        
        # 简化版本：不处理按钮事件，直接使用摇杆控制
        # 可以在这里添加紧急停止按钮等必要功能
        pass
    
    def get_command(self) -> Dict:
        """
        获取当前的控制命令
        
        Returns:
            包含控制信息的字典
        """
        if not self.is_connected:
            return {
                'x_velocity': 0.0,
                'y_velocity': 0.0,
                'angular_velocity': 0.0,
                'is_locked': False,
                'motor_state': 1
            }
        
        try:
            # 读取摇杆值
            # 左摇杆 - 线性运动
            left_x = self._apply_deadzone(self.joystick.get_axis(0))  # 左右
            left_y = self._apply_deadzone(-self.joystick.get_axis(1))  # 前后 (反向)
            
            # 右摇杆X轴 - 旋转
            right_x = self._apply_deadzone(self.joystick.get_axis(2))  # 旋转
            
            # 转换为速度命令
            x_velocity = left_y * self.max_linear_speed    # 前后
            y_velocity = left_x * self.max_linear_speed    # 左右
            angular_velocity = -right_x * self.max_angular_speed  # 旋转
            
            return {
                'x_velocity': x_velocity,
                'y_velocity': y_velocity,
                'angular_velocity': angular_velocity,
                'is_locked': self.is_locked,
                'motor_state': self.motor_state
            }
            
        except Exception as e:
            print(f"读取摇杆数据错误: {e}")
            return {
                'x_velocity': 0.0,
                'y_velocity': 0.0,
                'angular_velocity': 0.0,
                'is_locked': False,
                'motor_state': 1
            }
    
    def get_raw_input(self) -> Dict:
        """
        获取原始输入数据 (用于调试)
        
        Returns:
            包含原始输入数据的字典
        """
        if not self.is_connected:
            return {}
        
        try:
            raw_data = {
                'axes': [self.joystick.get_axis(i) for i in range(self.joystick.get_numaxes())],
                'buttons': [self.joystick.get_button(i) for i in range(self.joystick.get_numbuttons())],
                'hats': [self.joystick.get_hat(i) for i in range(self.joystick.get_numhats())]
            }
            return raw_data
        except Exception as e:
            print(f"获取原始数据错误: {e}")
            return {}
    
    def print_debug_info(self):
        """打印调试信息"""
        if not self.is_connected:
            print("游戏手柄未连接")
            return
        
        raw = self.get_raw_input()
        cmd = self.get_command()
        
        print(f"原始轴值: {[f'{x:.2f}' for x in raw.get('axes', [])]}")
        print(f"按钮状态: {raw.get('buttons', [])}")
        print(f"帽子状态: {raw.get('hats', [])}")
        print(f"速度命令: x={cmd['x_velocity']:.2f}, y={cmd['y_velocity']:.2f}, ω={cmd['angular_velocity']:.2f}")
        print(f"锁定状态: {cmd['is_locked']}, 电机状态: {cmd['motor_state']}")
        print("-" * 50)
    
    def cleanup(self):
        """清理资源"""
        if self.joystick and self.joystick.get_init():
            self.joystick.quit()
        pygame.joystick.quit()
        pygame.quit()
        print("北通游戏手柄控制器已清理")


def test_joystick():
    """测试游戏手柄功能"""
    print("开始测试北通游戏手柄...")
    
    try:
        controller = BeitongJoystickController(wait_timeout=10)
        
        print("测试开始，按Ctrl+C停止")
        while True:
            controller.update()
            controller.print_debug_info()
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\n测试被用户中断")
    except Exception as e:
        print(f"测试错误: {e}")
    finally:
        if 'controller' in locals():
            controller.cleanup()


if __name__ == "__main__":
    test_joystick() 