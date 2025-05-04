import os
import logging
from pathlib import Path
import anthropic
from elevenlabs import save
from elevenlabs.client import ElevenLabs
from tenacity import retry, stop_after_attempt, wait_exponential
import json
import warnings
# Only ignore the specific SyntaxWarning and RuntimeWarning from pydub.utils
warnings.filterwarnings("ignore", message=r".*invalid escape sequence.*", category=SyntaxWarning, module=r"pydub\.utils")
from pydub import AudioSegment

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class InterviewSimulator:
    def __init__(self):
        """Initialize the interview simulator with API clients."""
        # Load environment variables
        self.anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")
        self.elevenlabs_api_key = os.environ.get("ELEVENLABS_API_KEY")
        
        if not self.anthropic_api_key or not self.elevenlabs_api_key:
            raise ValueError("API keys for Anthropic and ElevenLabs are required.")
        
        # Initialize clients
        self.claude_client = anthropic.Anthropic(api_key=self.anthropic_api_key)
        self.elevenlabs_client = ElevenLabs(api_key=self.elevenlabs_api_key)
        
        # Voice IDs (customize these with your preferred voices)
        self.interviewer_voice_id = "UgBBYS2sOqTuMpoF3BR0"
        self.candidate_voice_id = "rYW2LlWtM70M5vc3HBtm"
        
    def load_context(self, resume_path, job_description_path):
        """Load resume and job description content."""
        logger.info(f"Loading resume from {resume_path}")
        with open(resume_path, 'r', encoding='utf-8') as file:
            resume_text = file.read()
        
        logger.info(f"Loading job description from {job_description_path}")
        with open(job_description_path, 'r', encoding='utf-8') as file:
            job_description = file.read()
            
        return resume_text, job_description
    
    def create_system_prompt(self, resume_text, job_description):
        """Create a detailed system prompt for Claude."""
        system_prompt = f"""You are an experienced technical interviewer conducting a job interview.

        <job_description>
        {job_description}
        </job_description>

        <candidate_resume>
        {resume_text}
        </candidate_resume>

        Your task is to conduct a realistic technical interview based on the job description and candidate's resume.
        Create a dialogue between "Interviewer" and "Candidate" with the following characteristics:
        
        1. Ask 5-7 relevant technical and behavioral questions
        2. Make questions specific to the candidate's experience and the job requirements
        3. Include thoughtful follow-up questions based on the candidate's responses
        4. Create realistic candidate responses based on their resume
        5. Include natural conversational elements like brief pauses, filler words, and occasional self-corrections
        6. Format the output with clear "Interviewer:" and "Candidate:" labels
        7. Begin with an introduction and end with a conclusion
        8. If the candidate does not know the answer, they should admit so honestly and not fabricate an answer.
        9. If the candidate does not know the answer, the interviewer should not press them for an answer.
        10. The candidate should give answers that they have indicated that they have knowledge of. Do not invent experience or skills that the candidate does not have which would not be universally known or a common skill/experience.

        The conversation should flow naturally and showcase both the interviewer's expertise and the candidate's skills.
        """
        
        return system_prompt
    
    def create_recruiter_phone_screen_system_prompt(self, resume_text, job_description):
        """Create a system prompt for a high-level candidate fit assessment."""
        system_prompt = f"""You are an experienced recruiter tasked with evaluating a candidate's overall fit for a technical role.

        <job_description>
        {job_description}
        </job_description>

        <candidate_resume>
        {resume_text}
        </candidate_resume>

        Your task is to provide a concise assessment of how well the candidate's background aligns with the job requirements.
        Include:
        1. A summary of the candidate's key strengths that match the core responsibilities.
        2. Any gaps or areas for development relative to the role.
        3. How specific skills and past achievements map to critical job requirements.
        4. An overall fit rating on a scale from 1 (poor fit) to 5 (excellent fit) with a brief rationale.

        Provide your evaluation in clear, professional language suitable for sharing with hiring managers."""
        
        return system_prompt
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def generate_interview(self, resume_text, job_description):
        """Generate an interview conversation using Claude API."""
        try:
            logger.info("Generating interview conversation with Claude")
            
            # Create system prompt
            system_prompt = self.create_system_prompt(resume_text, job_description)
            
            # Initial user message
            user_message = "Generate a realistic job interview conversation based on the provided resume and job description."
            
            # Generate the interview conversation
            response = self.claude_client.messages.create(
                model="claude-3-opus-20240229",
                max_tokens=4000,
                temperature=0.7,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_message}
                ]
            )
            
            # Extract the interview text
            interview_text = response.content[0].text
            return interview_text
            
        except anthropic.APIError as e:
            logger.error(f"Claude API error: {str(e)}")
            raise
    
    def process_conversation(self, interview_text):
        """Process the interview text into structured conversation turns."""
        import re
        
        logger.info("Processing conversation text into structured turns")
        
        # Extract dialogue turns with speaker labels
        pattern = r"(Interviewer:|Candidate:)\s*(.*?)(?=(?:Interviewer:|Candidate:)|$)"
        matches = re.findall(pattern, interview_text, re.DOTALL)
        
        conversation_turns = []
        for speaker, text in matches:
            speaker_role = speaker.strip(":")
            conversation_turns.append({
                "speaker": speaker_role,
                "text": text.strip()
            })
        
        logger.info(f"Extracted {len(conversation_turns)} conversation turns")
        return conversation_turns
    
    def format_text_for_speech(self, text, speaker):
        """Format text to improve speech naturalness."""
        
        # Add SSML break tags for natural pauses
        text = text.replace("...", " <break time=\"1s\"/> ")
        text = text.replace("—", " <break time=\"0.5s\"/> ")
        
        # Add pauses after commas and periods for more natural speech
        text = text.replace(", ", ", <break time=\"200ms\"/> ")
        text = text.replace(". ", ". <break time=\"500ms\"/> ")
        
        # Add emphasis to important words (customize based on content)
        if speaker == "Interviewer":
            # Interviewer tends to emphasize question words
            text = text.replace(" why ", " <emphasis>why</emphasis> ")
            text = text.replace(" how ", " <emphasis>how</emphasis> ")
            text = text.replace(" what ", " <emphasis>what</emphasis> ")
        else:
            # Candidate tends to emphasize achievements and skills
            text = text.replace(" successfully ", " <emphasis>successfully</emphasis> ")
            text = text.replace(" led ", " <emphasis>led</emphasis> ")
            text = text.replace(" created ", " <emphasis>created</emphasis> ")
        
        # Handle tech term pronunciations
        text = text.replace("SQL", "<say-as interpret-as=\"characters\">SQL</say-as>")
        text = text.replace("API", "<say-as interpret-as=\"characters\">API</say-as>")
        text = text.replace("CSS", "<say-as interpret-as=\"characters\">CSS</say-as>")
        
        return text
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def convert_turn_to_audio(self, turn, output_dir, index):
        """Convert a single conversation turn to audio."""
        speaker = turn["speaker"]
        text = turn["text"]
        
        # Format text for better speech
        formatted_text = self.format_text_for_speech(text, speaker)
        
        # Select voice based on speaker
        voice_id = self.interviewer_voice_id if speaker == "Interviewer" else self.candidate_voice_id
        
        # Generate audio
        logger.info(f"Generating audio for {speaker}, turn {index+1}")
        
        try:
            audio = self.elevenlabs_client.text_to_speech.convert(
                text=formatted_text,
                voice_id=voice_id,
                model_id="eleven_multilingual_v2",
                output_format="mp3_44100_128",
            )
            
            # Save audio file
            filename = f"{output_dir}/{index+1:02d}_{speaker}.mp3"
            save(audio, filename)
            
            logger.info(f"Saved audio to {filename}")
            return filename
            
        except Exception as e:
            logger.error(f"Error generating audio for turn {index+1}: {str(e)}")
            raise
    
    def convert_to_audio(self, conversation_turns, output_dir="interview_audio"):
        """Convert conversation turns to audio files."""
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        audio_files = []
        
        # Process each turn in the conversation
        for i, turn in enumerate(conversation_turns):
            filename = self.convert_turn_to_audio(turn, output_dir, i)
            audio_files.append(filename)
        
        return audio_files
    
    def run_simulation(self, resume_path, job_description_path, output_dir="output"):
        """Run the complete interview simulation process."""
        try:
            # Create output directory
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            
            # Load context
            resume_text, job_description = self.load_context(resume_path, job_description_path)
            
            # Generate interview
            interview_text = self.generate_interview(resume_text, job_description)
            
            # Save interview text
            text_output_path = f"{output_dir}/interview_transcript.txt"
            with open(text_output_path, "w", encoding="utf-8") as f:
                f.write(interview_text)
            logger.info(f"Saved interview transcript to {text_output_path}")
            
            # Process conversation
            conversation_turns = self.process_conversation(interview_text)
            
            # Convert to audio
            audio_output_dir = f"{output_dir}/audio"
            audio_files = self.convert_to_audio(conversation_turns, audio_output_dir)
            
            # Combine all audio files into one
            combined_audio = AudioSegment.empty()
            for audio_file in audio_files:
                combined_audio += AudioSegment.from_file(audio_file, format="mp3")
            combined_path = f"{output_dir}/combined_audio.mp3"
            combined_audio.export(combined_path, format="mp3")
            logger.info(f"Saved combined audio to {combined_path}")
            
            # Save manifest file with transcript, audio files, and combined audio
            manifest = {
                "transcript": text_output_path,
                "audio_files": audio_files,
                "combined_audio": combined_path
            }
            manifest_path = f"{output_dir}/manifest.json"
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2)
            logger.info(f"Saved manifest to {manifest_path}")
            return {
                "transcript": text_output_path,
                "audio_files": audio_files,
                "combined_audio": combined_path,
                "manifest": manifest_path
            }
            
        except Exception as e:
            logger.error(f"Error in interview simulation: {str(e)}")
            raise

# Example usage
if __name__ == "__main__":
    # Set up environment variables or load from .env
    from dotenv import load_dotenv
    load_dotenv()
    
    simulator = InterviewSimulator()
    
    result = simulator.run_simulation(
        resume_path="data/steve.txt",
        job_description_path="data/flyio.txt",
        output_dir="interview_output_steve_flyio"
    )

    print(f"Interview simulation completed!")
    print(f"Manifest: {result['manifest']}")
    print(f"Transcript: {result['transcript']}")
    print(f"Audio files: {result['audio_files']}")
    print(f"Combined audio: {result['combined_audio']}")