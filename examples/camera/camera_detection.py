import cv2
import time
import subprocess
import re

def get_camera_list():
    """
    Parses `v4l2-ctl` output to find all usable V4L2 video devices.

    Returns:
        A list of dictionaries, each representing a usable camera stream.
    """
    try:
        result = subprocess.run(
            ['v4l2-ctl', '--list-devices'],
            capture_output=True, text=True, check=True
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: `v4l2-ctl` command failed. Please ensure `v4l-utils` is installed.")
        return []

    usable_cameras = []
    format_priority = ['YUYV', 'MJPG']
    
    device_blocks = re.split(r'\n(?!\t)', result.stdout.strip())

    for block in device_blocks:
        if not block.strip() or '/dev/video' not in block:
            continue
        
        lines = block.strip().split('\n')
        camera_name = lines[0].split(':')[0]
        device_ports = re.findall(r'/dev/video(\d+)', block)

        # Find the first working port for this device
        for port in device_ports:
            device_path = f"/dev/video{port}"
            # Check which formats this port supports
            try:
                formats_output = subprocess.check_output(
                    ['v4l2-ctl', '-d', device_path, '--list-formats'],
                    text=True, stderr=subprocess.DEVNULL
                )
                
                # Check our preferred formats against the supported ones
                for fmt in format_priority:
                    if f"'{fmt}'" in formats_output:
                        usable_cameras.append({
                            'name': camera_name,
                            'port': int(port),
                            'format': fmt
                        })
                        break
                else: # This 'else' belongs to the 'for fmt' loop
                    continue # No desired format found, check the next port
                break # A format was found, so break from the 'for port' loop

            except subprocess.CalledProcessError:
                continue
    
    return usable_cameras

def show_camera_feed(camera_info, camera_index, total_cameras):
    """
    Opens and displays the video feed for a single camera.
    
    Args:
        camera_info (dict): A dictionary containing the camera's details.
        camera_index (int): The current camera's index for display purposes.
        total_cameras (int): The total number of cameras to be shown.
    """
    port = camera_info['port']
    fourcc_str = camera_info['format']
    name = camera_info['name']
    
    print(f"\nDisplaying Camera {camera_index + 1}/{total_cameras}: {name} on /dev/video{port}")

    cap = cv2.VideoCapture(port, cv2.CAP_V4L2)
    if not cap.isOpened():
        print(f"Error: Could not open video port {port}. Skipping.")
        return

    # Apply standard settings
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc_str))
    
    time.sleep(0.5) # Allow settings to apply

    window_title = f"[{camera_index + 1}/{total_cameras}] {name} (Port {port})"
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to retrieve frame.")
            time.sleep(0.5)
            continue
        
        cv2.imshow(window_title, frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    print(f"Closing feed for {name}.")
    cap.release()
    cv2.destroyAllWindows()
    time.sleep(0.1) # Brief pause to ensure window manager catches up

def main():
    """
    Finds all usable cameras and displays their streams sequentially.
    """
    cameras = get_camera_list()
    
    if not cameras:
        print("No usable video cameras found. Exiting.")
        return

    print(f"Found {len(cameras)} usable camera stream(s).")
    print("Press 'q' in the video window to cycle to the next camera.")

    for i, camera in enumerate(cameras):
        show_camera_feed(camera, i, len(cameras))

    print("\nAll camera streams have been shown. Exiting.")

if __name__ == '__main__':
    main()