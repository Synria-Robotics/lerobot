import cv2
import time
import platform
import subprocess
import re
import shutil

def default_backend():
    system_name = platform.system()
    if system_name == 'Windows':
        return cv2.CAP_DSHOW
    if system_name == 'Darwin':
        return cv2.CAP_AVFOUNDATION
    if system_name == 'Linux':
        return cv2.CAP_V4L2
    return cv2.CAP_ANY

def get_linux_cameras(backend):
    if shutil.which('v4l2-ctl') is None:
        return []

    try:
        result = subprocess.run(
            ['v4l2-ctl', '--list-devices'],
            capture_output=True,
            text=True,
            check=True
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
            try:
                formats_output = subprocess.check_output(
                    ['v4l2-ctl', '-d', device_path, '--list-formats'],
                    text=True,
                    stderr=subprocess.DEVNULL
                )

                for fmt in format_priority:
                    if f"'{fmt}'" in formats_output:
                        usable_cameras.append({
                            'name': camera_name,
                            'source': device_path,
                            'backend': backend,
                            'fourcc': fmt
                        })
                        break
                else:
                    continue

                break
            except subprocess.CalledProcessError:
                continue

    return usable_cameras

def get_camera_list(max_index=10, backend=None):
    """
    Probes camera indices using OpenCV and returns usable cameras.

    Args:
        max_index (int): The maximum camera index to probe.
        backend (int | None): OpenCV backend to use, defaults per platform.

    Returns:
        A list of dictionaries, each representing a usable camera stream.
    """
    if backend is None:
        backend = default_backend()

    if platform.system() == 'Linux':
        linux_cameras = get_linux_cameras(backend)
        if linux_cameras:
            return linux_cameras

    usable_cameras = []
    for index in range(max_index):
        cap = cv2.VideoCapture(index, backend)
        if not cap.isOpened():
            cap.release()
            continue

        ret, _ = cap.read()
        if not ret:
            cap.release()
            continue

        usable_cameras.append({
            'name': f'Camera {index}',
            'source': index,
            'backend': backend
        })
        cap.release()

    return usable_cameras

def show_camera_feed(camera_info, camera_index, total_cameras):
    """
    Opens and displays the video feed for a single camera.
    
    Args:
        camera_info (dict): A dictionary containing the camera's details.
        camera_index (int): The current camera's index for display purposes.
        total_cameras (int): The total number of cameras to be shown.
    """
    source = camera_info['source']
    name = camera_info['name']
    backend = camera_info.get('backend', default_backend())
    fourcc = camera_info.get('fourcc')

    if isinstance(source, int):
        source_label = f"index {source}"
    else:
        source_label = str(source)

    print(f"\nDisplaying Camera {camera_index + 1}/{total_cameras}: {name} on {source_label}")

    cap = cv2.VideoCapture(source, backend)
    if not cap.isOpened():
        print(f"Error: Could not open video source {source_label}. Skipping.")
        return

    # Apply standard settings
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if fourcc:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
    
    time.sleep(0.5) # Allow settings to apply

    window_title = f"[{camera_index + 1}/{total_cameras}] {name} ({source_label})"
    
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
