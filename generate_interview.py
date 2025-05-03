#!/usr/bin/env python3
"""
Generate a simulated interview using Anthropic Claude and convert to audio with ElevenLabs.

Usage:
    python generate_interview.py --input_file path/to/text.txt \
        --interviewer_voice "voice_name1" --interviewee_voice "voice_name2"

Ensure the environment variables ANTHROPIC_API_KEY and ELEVENLABS_API_KEY are set.
Requires ffmpeg installed for pydub.
"""
import os
import json
import argparse
from dotenv import load_dotenv
from anthropic import Client as AnthropicClient
from elevenlabs import generate, save
from pydub import AudioSegment

# Load environment variables from a .env file if present
load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")

if not ANTHROPIC_API_KEY or not ELEVENLABS_API_KEY:
    print("Please set ANTHROPIC_API_KEY and ELEVENLABS_API_KEY environment variables.")
    exit(1)

# Initialize clients
anthropic_client = AnthropicClient(api_key=ANTHROPIC_API_KEY)


def generate_interview_text(input_text: str) -> list:
    """
    Calls Claude to generate a simulated interview based on input_text.
    Returns a list of dicts: [{"speaker": "Interviewer"/"Interviewee", "text": "..."}, ...]
    """
    system_prompt = "You are a helpful assistant."
    user_prompt = (
        "You are generating a simulated interview between a hiring manager (label as 'Interviewer') "
        "and a candidate (label as 'Interviewee') based on the following input data.\n\n" + input_text +
        "\n\nOutput a JSON array of objects in the exact format:\n[ {\"speaker\": \"Interviewer\", \"text\": \"...\"}, {\"speaker\": \"Interviewee\", \"text\": \"...\"}, ... ]\n"
        "Do not output anything else."
    )

    response = anthropic_client.completions.create(
        model="claude-v1",
        prompt=system_prompt + "\n" + user_prompt,
        max_tokens_to_sample=1000,
        stop_sequences=["\n"]
    )
    raw = response.completion.strip()
    try:
        interview_data = json.loads(raw)
    except json.JSONDecodeError:
        print("Failed to parse JSON from Claude response:")
        print(raw)
        exit(1)
    return interview_data


def synthesize_audio(lines: list, interviewer_voice: str, interviewee_voice: str) -> list:
    """
    Generates TTS audio for each line and saves to temporary files.
    Returns a list of file paths in order.
    """
    audio_files = []
    for idx, line in enumerate(lines):
        speaker = line.get("speaker")
        text = line.get("text", "")
        voice = interviewer_voice if speaker.lower() == "interviewer" else interviewee_voice
        print(f"Synthesizing line {idx+1} [{speaker}]: {text}")
        audio = generate(text=text, voice=voice, api_key=ELEVENLABS_API_KEY)
        file_name = f"clip_{idx+1:03d}_{speaker}.mp3"
        save(audio, file_name)
        audio_files.append(file_name)
    return audio_files


def combine_audio(files: list, output_path: str = "interview.mp3"):
    """
    Concatenate a list of audio files (mp3) into one output file.
    """
    final = AudioSegment.empty()
    for file in files:
        segment = AudioSegment.from_file(file, format="mp3")
        final += segment
    print(f"Exporting final interview to {output_path}")
    final.export(output_path, format="mp3")


def main():
    parser = argparse.ArgumentParser(
        description="Generate a simulated interview and output as audio."
    )
    parser.add_argument("--input_file", type=str, help="Path to text file with input data.")
    parser.add_argument("--input_data", type=str, help="Input data as a string.")
    parser.add_argument(
        "--interviewer_voice", type=str, required=True,
        help="ElevenLabs voice name for the interviewer."
    )
    parser.add_argument(
        "--interviewee_voice", type=str, required=True,
        help="ElevenLabs voice name for the interviewee."
    )
    parser.add_argument(
        "--output", type=str, default="interview.mp3",
        help="Output file path for the combined audio."
    )
    args = parser.parse_args()

    if args.input_file:
        with open(args.input_file, 'r') as f:
            input_text = f.read()
    elif args.input_data:
        input_text = args.input_data
    else:
        print("Please provide either --input_file or --input_data.")
        exit(1)

    lines = generate_interview_text(input_text)
    clips = synthesize_audio(lines, args.interviewer_voice, args.interviewee_voice)
    combine_audio(clips, args.output)


if __name__ == "__main__":
    main()