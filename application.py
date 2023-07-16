import glob
import logging
import os

from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import ffmpeg
import librosa
import librosa.display
import soundfile as sf

from poor_tone_detector import PoorToneDetector

application = Flask(__name__)
application.config['UPLOAD_FOLDER'] = 'static/uploads/'

os.makedirs(application.config['UPLOAD_FOLDER'], exist_ok=True)

poor_tone_detector = PoorToneDetector()

application.logger.setLevel(logging.INFO)


def clean_upload_folder(max_files=10):
    files = glob.glob(os.path.join(application.config['UPLOAD_FOLDER'], '*'))
    files.sort(key=os.path.getmtime, reverse=True)

    while len(files) > max_files:
        os.remove(files[-1])
        files.pop()
    
    application.logger.info('cleaned upload folder')

def convert_ogg_to_wav(ogg_path, wav_path):
    stream = ffmpeg.input(ogg_path)
    stream = ffmpeg.output(stream, wav_path)
    ffmpeg.run(stream, overwrite_output=True)

    application.logger.info('converted ogg to wav')

def normalize_audio(file_path, sr=44100):
    y, _ = librosa.load(file_path, sr=sr)
    y = librosa.util.normalize(y)
    sf.write(file_path, y, sr, subtype='PCM_16')
    application.logger.info('normalized audio file')

def detect_mistakes(file_path, sr=44100, delta=0.12, th=0.5):
    y, sr = librosa.load(file_path, sr=sr)
    application.logger.info('loaded audio file')
    application.logger.info('start detecting poor tones')
    _, onsets, scores = poor_tone_detector.detect_poor_tone(y, sr, delta=delta, th=th)

    return onsets, scores

@application.route('/')
def index():
    return render_template('index.html')

@application.route('/run', methods=['POST'])
def api_run():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected for uploading'}), 400
    filename = secure_filename(file.filename)
    file_path = os.path.join(application.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)
    application.logger.info('file uploaded successfully')

    if filename.endswith('.ogg'):
        wav_filename = filename[:-4] + '.wav'
        wav_path = os.path.join(application.config['UPLOAD_FOLDER'], wav_filename)
        convert_ogg_to_wav(file_path, wav_path)
        
        file_path = wav_path
        filename = wav_filename

    clean_upload_folder(10)

    normalize_audio(file_path)

    onsets, scores = detect_mistakes(file_path)

    return jsonify({
        'filename': filename,
        'onsets': onsets,
        'scores': [score.tolist() for score in scores],
    })

@application.route('/api/load/<filename>', methods=['GET'])
def api_audio(filename):
    return send_from_directory(application.config['UPLOAD_FOLDER'], filename)


if __name__ == '__main__':
    application.run(host="0.0.0.0", port=5000, debug=True)