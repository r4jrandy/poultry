

Poultry Egg Counter
YOLOv8 Training Guide — Data Collection to Deployment
Accurate egg counting with your own camera footage


Why We Switched to YOLOv8
The original app used Hough Circle Transform — a basic shape detector. It failed because hens' bodies, heads, and combs are also round and whitish. YOLOv8 is a modern AI object detector that learns exactly what YOUR eggs look like from real footage of YOUR farm.

Method	Accuracy	Hens Confused?	Lighting Sensitive?
Hough Circles (old)	~50-60%	Yes — frequently	Very sensitive
YOLOv8 (new)	90-97%	No — learns the difference	Handles variation well

How the Training Process Works
You only need to do Steps 1-3 once (or when you want to retrain). After that, Step 4 is the app you run daily.

1	Extract frames from your videos
Pulls one image per second, skips blurry or duplicate frames

2	Label the images
Draw boxes around eggs using free LabelImg tool — takes 1-2 hours

3	Train the model
Run training script — produces best.pt weights file

4	Run the egg counter
Use best.pt with the new counting script — replaces old app

Step 1 — Collecting Good Training Data
How Many Images Do You Need?
Images Labeled	Expected Accuracy	Time to Label
50 images	~70% (okay for testing)	~30 minutes
100 images	~82% (usable)	~1 hour
200 images	~90% (good)	~2 hours
400+ images	~95%+ (excellent)	~4 hours

Tip: Quality matters more than quantity. 100 varied, well-lit images beat 400 frames of the same angle.

What Videos to Record
Record short 30-60 second clips of each of these situations:
•	Normal operation — eggs rolling on the belt in good lighting
•	Different times of day — morning vs afternoon vs artificial lighting
•	Different egg counts — rails with lots of eggs, some sparse
•	Eggs near hens — the trickiest case; capture this specifically
•	Slightly different camera angles — in case the camera shifts
•	Partially visible eggs at frame edges

Recording Tips
•	Mount camera above the conveyor belt — stable mount, not hand-held
•	Aim for at least 30fps so eggs do not blur during motion
•	Record in good lighting — but also include some dim-light footage
•	720p or 1080p resolution is ideal

Extracting Frames from Your Videos
Run the frame extraction script:
pip install opencv-python numpy
python step1_extract_frames.py --video myvideo.mp4 --out data/raw_frames

For a whole folder of videos:
python step1_extract_frames.py --folder videos/ --out data/raw_frames

Adjust the frame rate (default: 1 frame per second):
python step1_extract_frames.py --video myvideo.mp4 --out data/raw_frames --fps 2

Step 2 — Labeling Your Images
Installing LabelImg
pip install labelImg
labelImg

Alternative: Use Roboflow (https://roboflow.com) — free web-based labeling, no install needed. Easier for first-time users.

Setting Up LabelImg for YOLO Format
1.	Open LabelImg
2.	Click View and turn on Auto Save Mode
3.	Click the YOLO button in the left panel — must say YOLO, not PascalVOC
4.	Click Open Dir and select your data/raw_frames folder
5.	Click Change Save Dir and select the same folder

IMPORTANT: Make sure LabelImg is in YOLO format. Look for the YOLO button on the left panel and click it before you start labeling.

How to Label Eggs
6.	Press W to draw a bounding box
7.	Draw a tight rectangle around the egg — include the full egg
8.	When the label dialog appears, type egg and press Enter
9.	Label every egg visible in the frame — do not skip any
10.	Press D to go to the next image

Key	Action
W	Draw new bounding box
D	Next image
A	Previous image
Ctrl+S	Save manually
Del	Delete selected box
Ctrl+Z	Undo

Labeling Rules
•	Label EVERY visible egg — even partially visible ones at the edges
•	Draw boxes TIGHT around each egg — no empty space
•	Do NOT label hens, cages, or rails — only eggs
•	Do NOT label eggs hidden behind a hen
•	Cracked or damaged eggs should still be labeled as egg

Step 3 — Organizing Data and Training
Organize the Dataset
pip install pyyaml
python step2_organize_dataset.py --labeled data/raw_frames --out dataset/

This creates: dataset/train, dataset/val, dataset/test folders and a data.yaml config file.

Start Training
pip install ultralytics torch torchvision
python step3_train.py --data dataset/data.yaml

Option	Default	When to Change
--model yolov8n.pt	yolov8s.pt	Use nano for Raspberry Pi or slow PCs
--model yolov8m.pt	yolov8s.pt	Use medium if you have a GPU and want higher accuracy
--epochs 150	80	Increase if mAP is still improving at epoch 80
--batch 8	auto	Reduce to 8 or 4 if you get out of memory error

Training takes 30 minutes to 3 hours depending on your PC. Best weights are always auto-saved. You can stop early with Ctrl+C.

Understanding Your Training Results
mAP50 Score	Quality	Action
Below 0.70	Poor	Label 100+ more images, especially tricky cases
0.70 - 0.84	Moderate	Label 50 more images and re-train
0.85 - 0.92	Good	Ready to use; optionally add more data
0.93+	Excellent	Production ready

After Training — Finding Your Model
The best weights are saved at:
runs/egg_counter/weights/best.pt

Keep this file safe. It is your trained model.

Step 4 — Running the Egg Counter
python step4_test_and_count.py --weights runs/egg_counter/weights/best.pt --source 1

Option	Example	Description
--source	0 or 1 or 2	Camera index (0=built-in, 1=external USB camera)
--source	video.mp4	Test with a recorded video file first
--rows	4	Number of cage rows visible in frame
--line	0.5	Counting line position (0.0 to 1.0 across frame)
--conf	0.45	Confidence threshold — lower detects more, higher is stricter
--headless	(flag)	No display window — for Raspberry Pi or servers
--report-dir	my_reports/	Where to save CSV and JSON reports

If the model misses eggs, lower --conf to 0.35. If you see false detections, raise it to 0.55.

Improving the Model Over Time
When to Retrain
•	When you move the camera to a new angle
•	When accuracy drops after seasonal lighting changes
•	When you add more cage rows
•	Every 2-3 months for best results

Adding New Data Without Starting Over
Add new labeled images to your dataset folder and re-run:
python step2_organize_dataset.py --labeled data/raw_frames --out dataset/
python step3_train.py --data dataset/data.yaml

Adding New Object Classes
In the future you might want to detect broken eggs, double eggs, or full trays. Edit the CLASS_NAMES list in step2_organize_dataset.py, label the new class in LabelImg, and re-train.

Quick Reference — All Commands
First-time Setup
pip install opencv-python numpy ultralytics torch torchvision pyyaml labelImg

Daily Use (after training)
python step4_test_and_count.py --weights best.pt --source 1 --rows 4

Re-training with New Data
python step1_extract_frames.py --folder new_videos/ --out data/raw_frames
# Label new frames in LabelImg
python step2_organize_dataset.py --labeled data/raw_frames --out dataset/
python step3_train.py --data dataset/data.yaml

Check Model Accuracy
python evaluate_model.py --weights best.pt --data dataset/data.yaml

Files Included in This Package
File	Purpose	When to Run
step1_extract_frames.py	Extract images from videos	Before labeling
step2_organize_dataset.py	Split data into train/val/test sets	After labeling
step3_train.py	Train YOLOv8 model	After organizing data
step4_test_and_count.py	Run egg counter with trained model	Daily use
evaluate_model.py	Check mAP, precision, recall	After each training run
egg_counter.py	Original Hough-circle version (backup)	Not recommended


Generated for your poultry farm egg counting system
