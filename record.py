import airsim
import numpy as np
import cv2
import time
import os
import threading
from collections import deque

# НАСТРОЙКИ
SAVE_PATH = "dataset/run_1.npz"
DURATION = 90.0
CAMERA_DT = 0.05         # 20 герц
IMU_DT = 0.01            # 100 герц
HEIGHT = -2.0
SPEED = 2.0
SEGMENT_TIME = 4.0

# Шум IMU
IMU_NOISE_STD_ACC = 0.1
IMU_NOISE_STD_GYRO = 0.05

# глоб. буферы
imu_buffer = deque(maxlen=20000) 
imu_lock = threading.Lock()
stop_imu_thread = False


# Поток IMU отдельным клиентом
def imu_logger_thread():
    global stop_imu_thread
    
    imu_client = airsim.MultirotorClient()
    imu_client.confirmConnection()
    print(f"[IMU Thread] Connected & Started @ 100Hz")
    
    while not stop_imu_thread:
        try:
            imu = imu_client.getImuData()
            t = time.time()
            
            if hasattr(imu, 'linear_acceleration'):
                acc = imu.linear_acceleration
                gyro = imu.angular_velocity
            else:
                acc = imu.accelerometer
                gyro = imu.gyroscope
            
            # Шум
            noisy_acc = np.array([acc.x_val, acc.y_val, acc.z_val]) + \
                        np.random.normal(0, IMU_NOISE_STD_ACC, 3)
            noisy_gyro = np.array([gyro.x_val, gyro.y_val, gyro.z_val]) + \
                         np.random.normal(0, IMU_NOISE_STD_GYRO, 3)
            
            data = np.array([
                t, 
                noisy_acc[0], noisy_acc[1], noisy_acc[2],
                noisy_gyro[0], noisy_gyro[1], noisy_gyro[2]
            ], dtype=np.float32)
            
            with imu_lock:
                imu_buffer.append(data)
                
        except Exception as e:
            pass
            
        time.sleep(IMU_DT)
    
    imu_client.enableApiControl(False)


# сборщик данных
class DataCollector:
    def __init__(self):
        # Главный клиент
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        
        self.data = {
            "rgb": [], "pose": [], 
            "cmd": [], "time": [], "imu_windows": []
        }
        print("Collector Init")

    def get_image(self):
        try:
            responses = self.client.simGetImages([
                airsim.ImageRequest("", airsim.ImageType.Scene, False, False)
            ])

            if responses and len(responses) > 0 and responses[0].image_data_uint8:
                img_rgb = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
                img_rgb = img_rgb.reshape(responses[0].height, responses[0].width, 3)
                img_rgb = cv2.resize(img_rgb, (224, 224))
                img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
                return img_rgb
            else:
                return np.zeros((224, 224, 3), dtype=np.uint8)
        except Exception as e:
            print(f"Cam Error: {e}")
            return np.zeros((224, 224, 3), dtype=np.uint8)

    def get_pose(self):
        try:
            pose = self.client.simGetVehiclePose()
            return np.array([
                pose.position.x_val, pose.position.y_val, pose.position.z_val,
                pose.orientation.w_val, pose.orientation.x_val, pose.orientation.y_val, pose.orientation.z_val
            ], dtype=np.float32)
        except:
            return np.zeros(7, dtype=np.float32)

    def pop_imu_window(self, t_start, t_end):

        # Забираю замеры IMU за интервал [t_start, t_end] и удаляю сразу
        window = []
        indices_to_remove = []
        
        with imu_lock:
            # данные, попадающие в интервал
            for i, item in enumerate(imu_buffer):
                imu_t = item[0]
                if t_start <= imu_t <= t_end:
                    window.append(item[1:])  
                    indices_to_remove.append(i)
                elif imu_t > t_end:
                    break
            
            # стираю прочитанные данные
            for i in reversed(indices_to_remove):
                del imu_buffer[i]
            
            # защита от утечки памяти
            # если есть данные старше t_start - 1 сек, удаляю
            cleanup_threshold = t_start - 1.0
            cleanup_count = 0
            while len(imu_buffer) > 0 and imu_buffer[0][0] < cleanup_threshold:
                imu_buffer.popleft()
                cleanup_count += 1
            
        # возвращаю массив или заглушку
        if len(window) > 0:
            return np.array(window, dtype=np.float32)
        else:
            return np.zeros((1, 6), dtype=np.float32)

    def fly_square(self):
        global stop_imu_thread
        
        # запуск IMU потока
        imu_thread = threading.Thread(target=imu_logger_thread)
        imu_thread.daemon = True
        imu_thread.start()
        
        # взлет
        print("Takeoff")
        self.client.takeoffAsync().join()
        self.client.moveToZAsync(HEIGHT, 1).join()
        time.sleep(2)
        
        print("Recording")
        start_time = time.time()
        frame_count = 0
        last_frame_t = start_time
        
        segment = 0
        segment_start_time = start_time
        
        try:
            while time.time() - start_time < DURATION:
                loop_start = time.time()
                current_t = time.time()
                elapsed_total = current_t - start_time
                elapsed_segment = current_t - segment_start_time
                
                if elapsed_segment > SEGMENT_TIME:
                    segment = (segment + 1) % 4
                    segment_start_time = current_t
                    print(f"\n[+] Turning to Segment {segment}")
                
                # векторы скорости
                if segment == 0:   vx, vy, yaw = SPEED, 0.0, 0.0
                elif segment == 1: vx, vy, yaw = 0.0, SPEED, 90.0
                elif segment == 2: vx, vy, yaw = -SPEED, 0.0, 180.0
                else:              vx, vy, yaw = 0.0, -SPEED, -90.0
                
                # шум
                vx += np.random.normal(0, 0.2)
                vy += np.random.normal(0, 0.2)
                
                self.client.moveByVelocityAsync(
                    vx, vy, 0.0, CAMERA_DT,
                    drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
                    yaw_mode=airsim.YawMode(is_rate=False, yaw_or_rate=yaw)
                )
                
                # сбор данных юхуу
                img_rgb = self.get_image()
                pose = self.get_pose()
                imu_window = self.pop_imu_window(last_frame_t, current_t)
                
                self.data["rgb"].append(img_rgb)
                self.data["pose"].append(pose)
                self.data["cmd"].append([vx, vy, 0.0, yaw])
                self.data["time"].append(elapsed_total)
                self.data["imu_windows"].append(imu_window)
                
                frame_count += 1
                last_frame_t = current_t
                
                if frame_count % 50 == 0:
                    print(f"\r {elapsed_total:.1f}s | Seg: {segment} | Frames: {frame_count} | IMU: {len(imu_window)}", end="")
                
                # чекаю fps
                elapsed = time.time() - loop_start
                sleep_time = CAMERA_DT - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    
        except KeyboardInterrupt:
            print("\n Interrupted")
        except Exception as e:
            print(f"\n Error: {e}")
        finally:
            stop_imu_thread = True
            imu_thread.join()
            self.save_data(frame_count)
            self.land()

    def save_data(self, count):
        print(f"\n Saving ({count} frames)")
        os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
        
        np.savez_compressed(
            SAVE_PATH,
            rgb=np.array(self.data["rgb"], dtype=np.uint8),
            pose=np.array(self.data["pose"], dtype=np.float32),
            cmd=np.array(self.data["cmd"], dtype=np.float32),
            time=np.array(self.data["time"], dtype=np.float32),
            imu_windows=np.array(self.data["imu_windows"], dtype=object),
        )
        print(f"Saved: {SAVE_PATH}")
        
        if len(self.data["imu_windows"]) > 0:
            avg_samples = np.mean([len(w) for w in self.data["imu_windows"]])
            print(f"Avg IMU samples: {avg_samples:.1f}")
            
            # Проверка на баг
            if avg_samples > 100:
                print(f"Too many IMU samples. Buffer")
            elif avg_samples < 2:
                print(f"Too few IMU samples. IMU")
            else:
                print(f"correct")

    def land(self):
        print("Landing")
        self.client.hoverAsync().join()
        self.client.landAsync().join()
        self.client.armDisarm(False)
        self.client.enableApiControl(False)
        print("[+] Done.")

if __name__ == "__main__":
    collector = DataCollector()
    collector.fly_square()