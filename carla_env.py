import carla
import random
import numpy as np
import time
import cv2

class CarlaEnv:
    def __init__(self, host='localhost', port=2000):
        self.client = carla.Client(host, port)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        self.blueprint_library = self.world.get_blueprint_library()
        
        self.vehicle = None
        self.camera = None

    def reset(self):
        self.destroy()
        self.world.tick()
        return 1

    def setup_camera(self, camera_type='rgb', width=800, height=600, fov=90, 
                     x_offset=-5.0, z_offset=2.5, pitch=-15.0):
        if self.vehicle is None:
            print("No vehicle available to attach camera to!")
            return None
            
        # Get camera blueprint
        camera_bp = self.blueprint_library.find(f'sensor.camera.{camera_type}')
        camera_bp.set_attribute('image_size_x', str(width))
        camera_bp.set_attribute('image_size_y', str(height))
        camera_bp.set_attribute('fov', str(fov))
        
        # Set camera transform relative to vehicle
        camera_transform = carla.Transform(
            carla.Location(x=x_offset, z=z_offset),
            carla.Rotation(pitch=pitch)
        )
        
        # Spawn camera attached to vehicle
        self.camera = self.world.spawn_actor(
            camera_bp, 
            camera_transform, 
            attach_to=self.vehicle,
            attachment_type=carla.AttachmentType.Rigid
        )
        
        if self.camera is not None:
            print(f"{camera_type.upper()} camera attached to vehicle successfully!")
            print(f"Camera ID: {self.camera.id}")
            print(f"Camera position relative to vehicle: x={x_offset}, z={z_offset}")
            print(f"Camera pitch: {pitch} degrees")
            return self.camera
        else:
            print("Failed to spawn camera")
            return None

    def switch_to_camera_view(self):
        """Switch the spectator view to follow the camera position."""
        if self.camera is None:
            print("No camera available to switch view to!")
            return False
            
        # Get the spectator actor (the viewer in CARLA)
        spectator = self.world.get_spectator()
        
        # Set spectator to camera's transform
        camera_transform = self.camera.get_transform()
        spectator.set_transform(camera_transform)
        
        print("Switched to camera view! The CARLA window now shows the camera perspective.")
        return True
    
    def switch_to_vehicle_view(self, distance=10.0, height=5.0):
        """Switch spectator to a third-person view of the vehicle."""

        if self.vehicle is None:
            print("No vehicle available to view!")
            return False
            
        spectator = self.world.get_spectator()
        vehicle_transform = self.vehicle.get_transform()
        
        # Calculate position behind and above the vehicle
        vehicle_location = vehicle_transform.location
        vehicle_rotation = vehicle_transform.rotation
        
        # Move back and up relative to vehicle orientation
        import math
        yaw_rad = math.radians(vehicle_rotation.yaw)
        
        spectator_location = carla.Location(
            x=vehicle_location.x - distance * math.cos(yaw_rad),
            y=vehicle_location.y - distance * math.sin(yaw_rad),
            z=vehicle_location.z + height
        )
        
        # Look at the vehicle
        spectator_rotation = carla.Rotation(
            pitch=-20.0,  # Look down slightly
            yaw=vehicle_rotation.yaw,  # Same direction as vehicle
            roll=0.0
        )
        
        spectator_transform = carla.Transform(spectator_location, spectator_rotation)
        spectator.set_transform(spectator_transform)
        
        print("✓ Switched to third-person vehicle view!")
        return True
    
    def enable_continuous_camera_view(self):
        if self.camera is None:
            print("No camera available!")
            return False
        
        def sync_spectator_to_camera():
            if self.camera is not None:
                spectator = self.world.get_spectator()
                camera_transform = self.camera.get_transform()
                spectator.set_transform(camera_transform)
        
        # Store the callback function
        self._camera_sync_callback = sync_spectator_to_camera
        
        # Note: In a real implementation, you'd want to call this every tick
        # For now, we'll provide it as a manual function
        print("✓ Continuous camera view function is ready!")
        print("Call env.sync_spectator_to_camera() every step for continuous following")
        return True

    def step(self, action):
        # Apply throttle, steer, brake
        control = carla.VehicleControl(
            throttle=action[0],
            steer=action[1],
            brake=action[2]
        )
        self.vehicle.apply_control(control)

        self.world.tick()

        return None, 0.0, False, {}

    def destroy(self):
        # Handle vehicle
        if hasattr(self, 'vehicle') and self.vehicle is not None:
            try:
                self.vehicle.destroy()
            except RuntimeError:
                # Actor is already destroyed, ignore the error
                pass
            except AttributeError:
                # Vehicle doesn't exist, ignore
                pass
        
        # Handle camera
        if hasattr(self, 'camera') and self.camera is not None:
            try:
                self.camera.destroy()
            except RuntimeError:
                # Actor is already destroyed, ignore the error
                pass
            except AttributeError:
                # Camera doesn't exist, ignore
                pass
        
        # Reset the references to None
        self.vehicle = None
        self.camera = None

    def listen_to_camera(self):
        """Start listening to camera data and display it in a window."""
        if self.camera is not None:
            print("Starting camera feed...")
            self.camera.listen(lambda image: self._process_image(image))
        else:
            print("No camera available to listen to!")

    def _process_image(self, image):
        """Process and display camera image."""
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = np.reshape(array, (image.height, image.width, 4))  # BGRA
        rgb_image = array[:, :, :3][:, :, ::-1]  # Convert to RGB for display
        
        cv2.imshow("CARLA Camera Feed", rgb_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Camera window closed.")
            cv2.destroyAllWindows()

    def stop_camera_listening(self):
        """Stop the camera from listening and close the display window."""
        if self.camera is not None:
            self.camera.stop()
            cv2.destroyAllWindows()
            print("Camera stopped listening.")
        else:
            print("No camera to stop.")