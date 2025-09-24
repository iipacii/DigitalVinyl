import numpy as np
from PIL import Image, ImageFilter, ImageOps, ImageDraw, ImageFont, ImageEnhance, ImageChops
import os
import io
import struct
import math
import random
import colorsys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def ensure_square_image(image_path, output_path=None):
    """Make sure the image is square by cropping to a center square"""
    if output_path is None:
        output_path = image_path
    
    try:
        logger.info(f"Opening image: {image_path}")
        img = Image.open(image_path)
        width, height = img.size
        logger.info(f"Original image dimensions: {width}x{height}")
        
        # Make the image square by cropping to the center
        if width != height:
            size = min(width, height)
            logger.info(f"Cropping image to square size: {size}x{size}")
            left = (width - size) // 2
            top = (height - size) // 2
            right = left + size
            bottom = top + size
            img = img.crop((left, top, right, bottom))
            logger.info("Image cropped successfully")
        
        # Save with appropriate format based on file extension
        save_format = 'PNG'
        if output_path.lower().endswith('.jpg') or output_path.lower().endswith('.jpeg'):
            save_format = 'JPEG'
            if img.mode in ('RGBA', 'P'):
                img = img.convert('RGB')
        
        img.save(output_path, format=save_format)
        logger.info(f"Saved square image to: {output_path} using format: {save_format}")
        return output_path
    except Exception as e:
        logger.error(f"Error in ensure_square_image: {str(e)}")
        raise

def process_audio_file(audio_path, song_name=None, genre=None):
    """Read audio file data and prepare it for encoding"""
    try:
        logger.info(f"Processing audio file: {audio_path}")
        
        # Read the entire audio file as binary data
        with open(audio_path, 'rb') as f:
            audio_data = f.read()
        logger.info(f"Read {len(audio_data)} bytes of audio data")
        
        # Store file type/extension
        file_ext = os.path.splitext(audio_path)[1].lower().encode('utf-8')
        logger.info(f"Audio file extension: {file_ext.decode()}")
        
        # Handle song name
        if song_name is None:
            song_name = os.path.basename(audio_path)
            song_name = os.path.splitext(song_name)[0]
        logger.info(f"Using song name: {song_name}")
        
        # Encode song name and genre
        encoded_song_name = song_name.encode('utf-8')
        song_name_len = len(encoded_song_name)
        
        # Handle genre
        if genre is None:
            genre = "Unknown"
        encoded_genre = genre.encode('utf-8')
        genre_len = len(encoded_genre)
        logger.info(f"Using genre: {genre}")
        
        # Create header
        magic_bytes = b'AVNL'
        data_len = len(audio_data)
        ext_len = len(file_ext)
        
        header = (
            magic_bytes + 
            struct.pack('<I', data_len) + 
            struct.pack('B', ext_len) + 
            file_ext +
            struct.pack('<H', song_name_len) +
            encoded_song_name +
            struct.pack('B', genre_len) +
            encoded_genre
        )
        
        logger.info(f"Created header (size: {len(header)} bytes)")
        logger.info(f"Total data size: {len(header) + len(audio_data)} bytes")
        
        return header + audio_data
    except Exception as e:
        logger.error(f"Error in process_audio_file: {str(e)}")
        raise

def create_retro_effect(img, intensity=0.8):
    """
    Create a retro-style effect on the image without vignette or radial gradient
    
    Args:
        img: The original image
        intensity: How strong the effect should be (0.0 to 1.0)
    
    Returns:
        A new image with the simplified retro effect applied
    """
    width, height = img.size
    logger.info(f"Starting retro effect on {width}x{height} image...")
    
    # Create a new image with the same data
    result = img.copy()
    
    # 1. Add a vintage color grading effect
    logger.info("1/5: Applying vintage color grading...")
    enhancer = ImageEnhance.Contrast(result)
    result = enhancer.enhance(1.2)
    
    # Convert to numpy array for faster processing
    logger.info("Converting image to numpy array for faster processing...")
    img_array = np.array(result)
    
    logger.info("Adjusting color balance...")
    # Adjust color balance toward a retro palette (using numpy operations)
    img_array[:, :, 0] = np.clip(img_array[:, :, 0] * 1.15, 0, 255)  # Strengthen reds
    img_array[:, :, 2] = np.clip(img_array[:, :, 2] * 0.85, 0, 255)  # Reduce blues
    
    result = Image.fromarray(img_array)
    logger.info("Color grading complete")
    
    # 2. Add subtle scan lines (like on old CRT screens)
    logger.info("2/5: Adding scan lines effect...")
    scan_lines = np.zeros((height, width, 4), dtype=np.uint8)
    scan_lines[::4, :, 3] = 25  # Alpha channel every 4 pixels
    scan_lines = Image.fromarray(scan_lines)
    
    # Apply scan lines
    result = Image.alpha_composite(result, scan_lines)
    logger.info("Scan lines added")
    
    # 3. Add a subtle RGB shift (chromatic aberration)
    logger.info("3/5: Applying RGB shift effect...")
    if random.random() < 0.7:
        img_array = np.array(result)
        # Shift red channel right
        img_array[:-2, 2:, 0] = img_array[:-2, :-2, 0]
        # Shift blue channel left
        img_array[:-2, :-2, 2] = img_array[:-2, 2:, 2]
        result = Image.fromarray(img_array)
        logger.info("RGB shift applied")
    else:
        logger.info("RGB shift skipped (30% chance)")
    
    # 4. Add evenly distributed noise/grain
    logger.info("4/5: Generating and applying noise pattern...")
    noise = np.random.randint(-15, 15, (height, width)) * (np.random.random((height, width)) < 0.3)
    noise = np.stack([noise] * 3 + [np.zeros_like(noise)], axis=-1)
    noise = Image.fromarray(noise.astype(np.uint8))
    result = Image.alpha_composite(result, noise)
    logger.info("Noise effect complete")
    
    # 5. Add a subtle duotone effect
    logger.info("5/5: Creating duotone effect...")
    duotone_strength = 0.12
    retro_palettes = [
        [(255, 92, 205), (10, 186, 181)],   # Purple-teal
        [(255, 165, 0), (139, 69, 19)],     # Orange-brown
        [(38, 50, 56), (176, 190, 197)],    # Dark blue-grey
    ]
    highlight_color, shadow_color = random.choice(retro_palettes)
    logger.info(f"Selected palette: {highlight_color} to {shadow_color}")
    
    # Convert to grayscale using numpy
    logger.info("Converting to grayscale for duotone...")
    img_array = np.array(result)
    grayscale = np.dot(img_array[:, :, :3], [0.299, 0.587, 0.114])
    
    # Create duotone effect using vectorized operations
    logger.info("Applying duotone color mapping...")
    brightness = grayscale / 255.0
    duotone = np.zeros((height, width, 4), dtype=np.uint8)
    for i in range(3):
        duotone[:, :, i] = np.clip(
            shadow_color[i] * (1 - brightness) + highlight_color[i] * brightness,
            0, 255
        ).astype(np.uint8)
    duotone[:, :, 3] = 255
    
    # Blend the duotone effect with the original
    logger.info("Blending duotone effect with original image...")
    result = Image.blend(result, Image.fromarray(duotone), duotone_strength)
    
    logger.info("Retro effect processing complete!")
    return result

def add_stamp(img, stamp_path=None):
    """
    Add a stamp to the top right corner of the image
    
    Args:
        img: The image to add the stamp to
        stamp_path: Path to the stamp image file
    
    Returns:
        The image with the stamp added
    """
    if stamp_path is None:
        # Use the stamp in the app directory by default
        current_dir = os.path.dirname(os.path.abspath(__file__))
        stamp_path = os.path.join(current_dir, "Stamp.png")
    
    if not os.path.exists(stamp_path):
        logger.warning(f"Stamp file not found at {stamp_path}")
        return img
    
    # Load the stamp image
    try:
        stamp = Image.open(stamp_path).convert('RGBA')
        
        # Calculate appropriate stamp size (15-20% of the image width)
        img_width, img_height = img.size
        stamp_scale = min(img_width, img_height) * 0.18 / max(stamp.size)
        stamp_width = int(stamp.size[0] * stamp_scale)
        stamp_height = int(stamp.size[1] * stamp_scale)
        
        # Resize the stamp
        stamp = stamp.resize((stamp_width, stamp_height), Image.LANCZOS)
        
        # Calculate position (top right with some padding)
        padding = int(min(img_width, img_height) * 0.03)  # 3% padding
        position = (img_width - stamp_width - padding, padding)
        
        # Create a copy of the original image
        result = img.copy()
        
        # Paste the stamp onto the image
        result.paste(stamp, position, stamp)
        
        return result
    except Exception as e:
        logger.error(f"Error adding stamp: {str(e)}")
        return img

def encode_audio_to_image(audio_path, cover_image_path, output_image_path, song_name=None, genre=None):
    """Encode audio data into an image"""
    try:
        logger.info("Starting audio encoding process")
        logger.info(f"Audio: {audio_path}")
        logger.info(f"Cover image: {cover_image_path}")
        logger.info(f"Output: {output_image_path}")
        logger.info(f"Genre: {genre}")
        
        # Ensure the cover image is square
        ensure_square_image(cover_image_path)
        
        # Process audio file
        try:
            data_to_encode = process_audio_file(audio_path, song_name, genre)
            logger.info(f"Processed audio data size: {len(data_to_encode)} bytes")
        except Exception as e:
            logger.error("Error processing audio file")
            raise
        
        # Read and process cover image
        try:
            cover_img = Image.open(cover_image_path).convert('RGBA')
            width, height = cover_img.size
            logger.info(f"Cover image dimensions: {width}x{height}")
            
            # Calculate maximum data capacity
            max_bytes = width * height * 3 // 8
            logger.info(f"Image capacity: {max_bytes} bytes")
            
            if len(data_to_encode) > max_bytes:
                logger.warning(f"Audio data ({len(data_to_encode)} bytes) exceeds image capacity ({max_bytes} bytes)")
                logger.info("Will encode truncated audio data")
        except Exception as e:
            logger.error("Error processing cover image")
            raise
        
        # Apply visual effects
        try:
            logger.info("Applying retro effect to image")
            stylized_img = create_retro_effect(cover_img)
            logger.info("Applying stamp to image")
            stylized_img = add_stamp(stylized_img)
        except Exception as e:
            logger.error("Error applying visual effects")
            raise
        
        # Encode the data
        try:
            logger.info("Starting data encoding process")
            pixels = np.array(stylized_img, dtype=np.uint8)
            flat_pixels = pixels.reshape(-1, 4)
            
            # Convert data to bits
            data_bits = []
            for byte in data_to_encode:
                bits = [int(b) for b in format(byte if isinstance(byte, int) else ord(byte), '08b')]
                data_bits.extend(bits)
            
            # Add terminator
            terminator = [1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1]
            data_bits.extend(terminator)
            
            logger.info(f"Total bits to encode: {len(data_bits)}")
            
            # Encode bits into pixels
            encoded_count = 0
            for i in range(min(len(data_bits), len(flat_pixels) * 3)):
                channel_idx = i % 3
                pixel_idx = i // 3
                
                pixel_val = int(flat_pixels[pixel_idx, channel_idx])
                if data_bits[i] == 1:
                    new_val = (pixel_val | 1)
                else:
                    new_val = (pixel_val & 254)
                new_val = max(0, min(255, new_val))
                flat_pixels[pixel_idx, channel_idx] = new_val
                encoded_count += 1
            
            logger.info(f"Successfully encoded {encoded_count} bits")
            
            # Reshape and save
            encoded_pixels = flat_pixels.reshape(pixels.shape)
            result_img = Image.fromarray(encoded_pixels, mode='RGBA')
            
            # Save with appropriate format
            output_ext = os.path.splitext(output_image_path)[1].lower()
            if output_ext in ['.jpg', '.jpeg']:
                result_img = result_img.convert('RGB')
                result_img.save(output_image_path, format="JPEG", quality=95)
            else:
                result_img.save(output_image_path)
                
            logger.info(f"Successfully saved encoded image to: {output_image_path}")
            
        except Exception as e:
            logger.error(f"Error during data encoding: {str(e)}")
            raise
        
        return output_image_path
        
    except Exception as e:
        logger.error(f"Error in encode_audio_to_image: {str(e)}")
        raise

def decode_audio_from_image(encoded_image_path, output_audio_path):
    """Decode audio data from an encoded image"""
    try:
        logger.info(f"Starting decoding from image: {encoded_image_path}")
        
        # Read the encoded image
        encoded_img = Image.open(encoded_image_path).convert('RGBA')
        width, height = encoded_img.size
        logger.info(f"Image dimensions: {width}x{height}")
        
        # Extract pixel data
        pixels = np.array(encoded_img, dtype=np.uint8)
        flat_pixels = pixels.reshape(-1, 4)
        
        # Extract bits
        extracted_bits = []
        for pixel in flat_pixels:
            for channel_idx in range(3):
                extracted_bits.append(int(pixel[channel_idx]) & 1)
        
        logger.info(f"Extracted {len(extracted_bits)} bits")
        
        # Convert bits to bytes
        extracted_bytes = bytearray()
        for i in range(0, len(extracted_bits) - 7, 8):
            if i + 8 <= len(extracted_bits):
                byte = 0
                for j in range(8):
                    byte = (byte << 1) | extracted_bits[i + j]
                extracted_bytes.append(byte)
        
        logger.info(f"Converted to {len(extracted_bytes)} bytes")
        
        # Look for magic bytes
        magic_bytes = b'AVNL'
        magic_pos = extracted_bytes.find(magic_bytes)
        
        song_name = None
        genre = None
        file_ext = ".mp3"  # Default extension
        
        if magic_pos == -1:
            logger.warning("Magic bytes not found, trying fallback decoding")
            # Fallback decoding logic...
        else:
            logger.info("Found magic bytes, parsing header")
            try:
                # Parse header
                header_start = magic_pos
                data_len_pos = header_start + 4
                data_len = struct.unpack('<I', extracted_bytes[data_len_pos:data_len_pos+4])[0]
                
                ext_len_pos = data_len_pos + 4
                ext_len = extracted_bytes[ext_len_pos]
                
                ext_pos = ext_len_pos + 1
                file_ext = extracted_bytes[ext_pos:ext_pos+ext_len].decode('utf-8')
                logger.info(f"Found file extension: {file_ext}")
                
                # Extract song name
                song_name_len_pos = ext_pos + ext_len
                song_name_len = struct.unpack('<H', bytes(extracted_bytes[song_name_len_pos:song_name_len_pos+2]))[0]
                
                if 0 < song_name_len < 500:
                    song_name_pos = song_name_len_pos + 2
                    song_name = extracted_bytes[song_name_pos:song_name_pos+song_name_len].decode('utf-8')
                    logger.info(f"Found song name: {song_name}")
                    
                    # Extract genre after song name
                    genre_len_pos = song_name_pos + song_name_len
                    if genre_len_pos < len(extracted_bytes):
                        genre_len = extracted_bytes[genre_len_pos]
                        genre_pos = genre_len_pos + 1
                        if genre_pos + genre_len <= len(extracted_bytes):
                            genre = extracted_bytes[genre_pos:genre_pos+genre_len].decode('utf-8')
                            logger.info(f"Found genre: {genre}")
                            audio_data_pos = genre_pos + genre_len
                        else:
                            audio_data_pos = song_name_pos + song_name_len
                    else:
                        audio_data_pos = song_name_pos + song_name_len
                else:
                    audio_data_pos = ext_pos + ext_len
                
                # Extract audio data
                audio_data = extracted_bytes[audio_data_pos:audio_data_pos+data_len]
                logger.info(f"Extracted {len(audio_data)} bytes of audio data")
                
            except Exception as e:
                logger.error(f"Error parsing header: {str(e)}")
                # Fallback to basic extraction
                audio_data = extracted_bytes[header_start + 20:]
        
        # Ensure valid file extension
        if not file_ext or not file_ext.startswith('.'):
            file_ext = ".mp3"
        file_ext = file_ext.lower()
        
        # Update output path with correct extension
        base_path, current_ext = os.path.splitext(output_audio_path)
        if current_ext.lower() != file_ext:
            output_audio_path = base_path + file_ext
        
        # Write audio data
        with open(output_audio_path, 'wb') as f:
            f.write(audio_data)
        
        logger.info(f"Successfully saved decoded audio to: {output_audio_path}")
        
        return output_audio_path, song_name, genre
        
    except Exception as e:
        logger.error(f"Error in decode_audio_from_image: {str(e)}")
        raise