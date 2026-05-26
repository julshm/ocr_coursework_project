@echo off
python main.py --input input_images/sample.jpg --output output --lang ukr --mode auto --ground-truth ground_truth/sample.txt
python main.py --input input_images/doc_scan3.jpg --output output --lang ukr --mode auto --ground-truth ground_truth/doc_scan3.txt
python main.py --input input_images/kvytancia.png --output output --lang ukr --mode auto --ground-truth ground_truth/kvytancia.txt
python main.py --input input_images/noisy_photo_coursework.png --output output --lang ukr+eng --mode auto --ground-truth ground_truth/noisy_photo_coursework.txt
python main.py --input input_images/noisy_photo_coursework_hard.png --output output --lang ukr+eng --mode auto --ground-truth ground_truth/noisy_photo_coursework_hard.txt
pause
