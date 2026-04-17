import cv2
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
import argparse

# ---------------------------------------------------------
# 1. Imports / Klassen
# ---------------------------------------------------------
try:
    from pose_estimation_recognition_utils_rtmlib import RTMPoseEstimator2D
    from pose_estimation_recognition_utils_rtmlib import RTMPoseEstimationFrom3DFrame
except ImportError as e:
    print(f"WARNUNG: Externe Abhängigkeiten konnten nicht importiert werden: {e}")
    # Für Entwicklungszwecke könnten hier Mocks definiert werden
    pass

class DatasetGenerator:
    def __init__(self, source_dir, output_dir_small, output_dir_medium, baseline_m=0.06):
        self.source_dir = Path(source_dir)
        self.output_dir_small = Path(output_dir_small)
        self.output_dir_medium = Path(output_dir_medium)
        self.output_dir_full = Path(output_dir_full) if output_dir_full else None
        self.baseline = baseline_m
        
        # Internal counters for naming files
        self.cnt_small = 0
        self.cnt_medium = 0
        self.cnt_full = 0
        
        # Constants from user
        self.CX_LEFT = 640
        self.CY_LEFT = 360
        self.FOCAL_LENGTH = 2710
        self.DISTANCE = 60 # Baseline

        # Initialisierung der Tools
        print("Initialisiere Pose Estimators...")
        try:
            # 1. Standard 2D Estimator (Explicitly requested by user)
            self.pose_estimator = RTMPoseEstimator2D(
                mode='individual', 
                det_model_path='https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_m_8xb8-300e_humanart-c2c7a14a.zip', 
                det_input_size=(640, 640),
                pose_model_path='https://cobtras.com/data/model.onnx',
                pose_input_size=(288, 384),
                device="cuda")
            
            # 2. 3D Lifter (Replacement for SAD)
            self.lifter = RTMPoseEstimationFrom3DFrame(
                focal_length=self.FOCAL_LENGTH,
                distance=self.DISTANCE,
                cx_left=self.CX_LEFT,
                cy_left=self.CY_LEFT,
                with_confidence=True,
                device="cuda",
                mode = 'individual',
                det_model_path='https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_m_8xb8-300e_humanart-c2c7a14a.zip', 
                det_input_size=(640, 640),
                pose_model_path='https://cobtras.com/data/model.onnx',
                pose_input_size=(288, 384)
            )
        except NameError:
             print("Fehler: Klassen nicht definiert. Import fehlgeschlagen?")

    def scan_videos(self):
        """Findet alle Videodateien im Quellverzeichnis."""
        extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv']
        video_files = []
        for ext in extensions:
            video_files.extend(list(self.source_dir.rglob(ext)))
        return sorted(video_files)

    def count_total_frames(self, video_files):
        """Zählt die Gesamtzahl der Frames in allen Videos."""
        total_frames = 0
        video_frames_map = {}
        
        print("Zähle Frames in allen Videos...")
        for video_path in tqdm(video_files):
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                print(f"Warnung: Kann Video nicht öffnen: {video_path}")
                continue
            
            n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            video_frames_map[video_path] = n_frames
            total_frames += n_frames
            cap.release()
            
        return total_frames, video_frames_map

    def process_frame(self, frame):
        """
        Verarbeitet einen Frame:
        1. 2D: PoseEstimation Left + Right -> Mean (Mittelwert).
        2. 3D: RTMPoseEstimationFrom3DFrame.
        """
        
        # A. 3D Extraction (Using Stereo Frame)
        try:
            results_3d = self.lifter.extract_frame(frame)
        except ZeroDivisionError:
             return None
        except Exception as e:
             print(f"Frame processing error: {e}")
             return None
             
        if not results_3d:
            # Wenn 3D fehlschlägt, geben wir meist auch nichts zurück, 
            # da das Dataset Lifting trainieren soll.
            return None

        # B. 2D Extraction (Explicit Mean per User Request)
        h, w, _ = frame.shape
        w_half = w // 2
        frame_left = frame[:, :w_half]
        frame_right = frame[:, w_half:]
        
        # Run Estimator on both sides
        res_left = self.pose_estimator.process_image(frame_left)
        res_right = self.pose_estimator.process_image(frame_right)
        
        # Helper to extract raw keypoints from result list
        def get_kps(res):
            if res and res.num_persons > 0:
                if isinstance(res, dict):
                    return res.get('keypoints')
                elif hasattr(res, 'keypoints'):
                    return res.keypoints
            return None

        kps_left = get_kps(res_left)
        kps_right = get_kps(res_right)
        
        # Helper for scores
        def get_scores(res):
            if res and res.num_persons > 0:
                if isinstance(res, dict):
                    return res.get('scores') or res.get('keypoint_scores')
                elif hasattr(res, 'scores'):
                    return res.scores
                elif hasattr(res, 'keypoint_scores'):
                    return res.keypoint_scores
            return None

        scores_left = get_scores(res_left)
        scores_right = get_scores(res_right)

        # Build Output Dictionaries
        kp2d_dict = {}
        kp3d_dict = {}
        
        # 1. Calculate 2D Mean
        if kps_left is not None and kps_right is not None:
             # kps_left shape is (N_people, N_joints, 2). N_people=1 usually.
             
             # Determine number of joints. Use shape[1] if available.
             if hasattr(kps_left, 'shape') and len(kps_left.shape) >= 2:
                 n_points = kps_left.shape[1]
             else:
                 n_points = len(kps_left[0])

             for i in range(n_points):
                 x_avg = (kps_left[0][i][0] + kps_right[0][i][0]) / 2.0
                 y_avg = (kps_left[0][i][1] + kps_right[0][i][1]) / 2.0
                 kp2d_dict[str(i)] = {"x": float(x_avg), "y": float(y_avg)}
                 
        elif kps_left is not None:
             # Fallback Left
             for i, p in enumerate(kps_left[0]):
                 kp2d_dict[str(i)] = {"x": float(p[0]), "y": float(p[1])}
                 
        elif kps_right is not None:
             # Fallback Right
             for i, p in enumerate(kps_right[0]):
                 kp2d_dict[str(i)] = {"x": float(p[0]), "y": float(p[1])}
        else:
            # Kein 2D gefunden -> Skip Frame
            return None

        # 2. Process 3D Keypoints
        for p in results_3d:
            kp3d_dict[str(p.get_data()["id"])] = {"x": float(p.get_data()["x"]), "y": float(p.get_data()["y"]), "z": float(p.get_data()["z"])}
            
        # Confidence
        # Prefer confidence from 3D lifter (which likely merges usage)
        max_id = 0
        if results_3d:
            max_id = max(max_id, max(p.get_data()["id"] for p in results_3d))
            
        if kp2d_dict:
            max_id = max(max_id, max(int(k) for k in kp2d_dict.keys()))

        conf_array = np.zeros(max_id + 1)
        
        # Prefer 3D confidence (merged), fallback to 2D
        
        # 3D
        for p in results_3d:
             data = p.get_data()
             if "confidence" in data:
                 idx = data["id"]
                 if idx < len(conf_array):
                     conf_array[idx] = float(data["confidence"])
        
        # Fallback 2D if 0
        if scores_left is not None:
            # scores_left shape usually (1, 133)
            # Flatten or access first person
            s_left = scores_left[0] if len(scores_left.shape) > 1 else scores_left
            for i in range(min(len(s_left), len(conf_array))):
                if conf_array[i] == 0:
                    conf_array[i] = float(s_left[i])
        
        # Fallback 2D right if still 0 and left was not available or had 0
        if scores_right is not None:
            s_right = scores_right[0] if len(scores_right.shape) > 1 else scores_right
            for i in range(min(len(s_right), len(conf_array))):
                if conf_array[i] == 0:
                    conf_array[i] = float(s_right[i])

        # print("1 frame ready") # Spam reduce
                
        return {
            "keypoints_2d": kp2d_dict,
            "keypoints_3d": kp3d_dict,
            "confidence": conf_array,
            "intrinsics": self.FOCAL_LENGTH
        }

    def run(self, target_small=25000, target_medium=50000):

        video_files = self.scan_videos()
        total_frames_all_videos, video_frames_map = self.count_total_frames(video_files)
        
        print(f"Gefunden: {len(video_files)} Videos mit insgesamt {total_frames_all_videos} Frames.")
        
        # 3. Sampling Intervalle
        step_small = max(1, total_frames_all_videos // target_small)
        step_medium = max(1, total_frames_all_videos // target_medium)
        
        print(f"Sampling Step Small: {step_small}")
        print(f"Sampling Step Medium: {step_medium}")
        
        global_frame_counter = 0
        processed_count = 0
        
        # Ensure output directories exist
        self.output_dir_small.mkdir(parents=True, exist_ok=True)
        self.output_dir_medium.mkdir(parents=True, exist_ok=True)
        if self.output_dir_full:
            self.output_dir_full.mkdir(parents=True, exist_ok=True)
        
        print(f"Output Directory Small: {self.output_dir_small}")
        print(f"Output Directory Medium: {self.output_dir_medium}")
        if self.output_dir_full:
            print(f"Output Directory Full: {self.output_dir_full}")
        
        # Check for existing frames to resume
        existing_frames = self._check_video_and_frameid()

        for video_path in video_files:
            try:
                cap = cv2.VideoCapture(str(video_path))
                n_frames = video_frames_map[video_path]
            
                pbar = tqdm(total=n_frames, desc=f"Verarbeite {video_path.name}")
            
                for f_idx in range(n_frames):
                    ret, frame = cap.read()
                    if not ret:
                        break
                
                    # Check ob wir diesen Frame für eines der Datasets brauchen
                    need_for_small = (global_frame_counter % step_small == 0)
                    need_for_medium = (global_frame_counter % step_medium == 0)
                    need_for_full = (self.output_dir_full is not None) # Save all to full
                
                    if need_for_small or need_for_medium or need_for_full:
                        # Check if already processed
                        if (video_path.name, global_frame_counter) in existing_frames:
                            global_frame_counter += 1
                            pbar.update(1)
                            continue

                        result = self.process_frame(frame)
                    
                        if result:
                            # Qualitätsprüfung
                            if np.mean(result['confidence']) < 0.5:
                                # Skip schlechte Frames
                                pass 
                            else:
                                # Speichern
                                self._save_result(result, global_frame_counter, need_for_small, need_for_medium, need_for_full, video_path.name)
                                processed_count += 1
                
                    global_frame_counter += 1
                    pbar.update(1)
            
                pbar.close()
                cap.release()

            except:
                print(f'Failed to read file {video_path}')
            
        print(f"Fertig! Verarbeitete Samples (gesamt verarbeitet): {processed_count}")
        print(f"Gespeicherte Samples Small: {self.cnt_small}")
        print(f"Gespeicherte Samples Medium: {self.cnt_medium}")
        if self.output_dir_full:
            print(f"Gespeicherte Samples Full: {self.cnt_full}")

    def _check_video_and_frameid(self):
        """
        Scannt die Output-Ordner nach existierenden Dateien, um:
        1. Die Counter (cnt_small, cnt_medium) korrekt zu setzen.
        2. Eine Menge von bereits verarbeiteten (video_name, frame_id) zu erstellen.
        """
        processed_frames = set()
        
        def scan_dir(dir_path, is_small):
            max_cnt = -1
            if not dir_path.exists():
                return
            
            # Wir suchen nach pattern *.json
            # Achtung: Wenn es sehr viele Dateien sind, kann das dauern.
            json_files = list(dir_path.glob("*.json"))
            if not json_files:
                return

            print(f"Scanne {len(json_files)} Dateien in {dir_path}...")
            
            for fpath in tqdm(json_files, desc=f"Scanning {dir_path.name}"):
                # 1. Counter update
                try:
                    # Dateiname ist z.B. 00000123.json
                    file_idx = int(fpath.stem)
                    if file_idx > max_cnt:
                        max_cnt = file_idx
                except ValueError:
                    pass
                
                # 2. Content check (optional, aber sicherer für Duplikatsvermeidung)
                # Wir lesen nur kurz den Header infos, wenn wir wirklich sicher gehen wollen,
                # dass wir GENAU diesen Frame skippen.
                # Performance-Optimierung: Ggf. nur Video/Frame Namen aus Cache nutzen? 
                # Hier: Einmalig einlesen.
                try:
                    with open(fpath, 'r') as f:
                        data = json.load(f)
                        src_vid = data.get("source_video")
                        fid = data.get("frame_id")
                        if src_vid is not None and fid is not None:
                            processed_frames.add((src_vid, int(fid)))
                except Exception as e:
                    print(f"Fehler beim Lesen von {fpath}: {e}")

            # Setze Counter auf max + 1
            if is_small:
                self.cnt_small = max_cnt + 1
            else:
                self.cnt_medium = max_cnt + 1

        scan_dir(self.output_dir_small, is_small=True)
        scan_dir(self.output_dir_medium, is_small=False)
        if self.output_dir_full:
            scan_dir(self.output_dir_full, is_small=False) # Reuse False or update logic

        print(f"Bereits verarbeitet: {len(processed_frames)} eindeutige Frames (aus Small & Medium).")
        print(f"Fortfahren bei Index Small: {self.cnt_small}")
        print(f"Fortfahren bei Index Medium: {self.cnt_medium}")
        
        return processed_frames
         


    def _save_result(self, result, frame_id, save_small, save_medium, save_full, source_video):
        # Result is already in dict format from process_frame
        
        packet = {
            "keypoints_2d": result['keypoints_2d'],
            "keypoints_3d": result['keypoints_3d'],
            "source_video": source_video,
            "frame_id": int(frame_id),
            "confidence": result['confidence'].tolist(),
             "meta": {
                "camera": {
                  "baseline_m": self.baseline,
                  "intrinsics": float(result['intrinsics'])
                }
            }
        }

        # Dateiname: <index>_<video>_<frame>.json
        # Wir nutzen den globalen counter als Index
        
        if save_small:
            file_name = f"{self.cnt_small:08d}.json" 
            out_path = self.output_dir_small / file_name
            with open(out_path, 'w') as f:
                json.dump(packet, f, indent=2)
            self.cnt_small += 1
        
        if save_medium:
            file_name = f"{self.cnt_medium:08d}.json"
            out_path = self.output_dir_medium / file_name
            with open(out_path, 'w') as f:
                json.dump(packet, f, indent=2)
            self.cnt_medium += 1

        if save_full and self.output_dir_full:
            file_name = f"{self.cnt_full:08d}.json"
            out_path = self.output_dir_full / file_name
            with open(out_path, 'w') as f:
                json.dump(packet, f, indent=2)
            self.cnt_full += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stereo Dataset Generator")
    parser.add_argument("--source", type=str, required=True, help="Pfad zu den Videos")
    parser.add_argument("--out_small", type=str, default="dataset_small_cache", help="Output Verzeichnis Small (Ordner)")
    parser.add_argument("--out_medium", type=str, default="dataset_medium_cache", help="Output Verzeichnis Medium (Ordner)")
    parser.add_argument("--out_full", type=str, default=None, help="Output Verzeichnis Full (Ordner, optional)")
    
    args = parser.parse_args()
    
    generator = DatasetGenerator(args.source, args.out_small, args.out_medium, args.out_full)
    generator.run()
