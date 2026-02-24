import telebot
from telebot import types
import asyncio
import aiohttp
import logging
import re
import io
import threading
from PIL import Image
import tempfile
import os
import speech_recognition as sr
from pydub import AudioSegment
import base64

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Bot token
TOKEN = os.environ.get("BOT_TOKEN", "YOUR_BOT_TOKEN_HERE")
bot = telebot.TeleBot(TOKEN)

# API settings
API_TOKEN = os.environ.get("API_TOKEN", "YOUR_API_TOKEN_HERE")
API_URL = "https://api.intelligence.io.solutions/api/v1/chat/completions"
DOWNLOADER_API_URL = "https://download.dreampartners.online"

# Main system prompt (same as app.py)
MAIN_SYSTEM_PROMPT = "Отвечай кратко и по делу. Избегай лишних слов и длинных объяснений, если не требуется детальный ответ."

# Special system prompt for dreamGPT AI
DREAMGPT_SYSTEM_PROMPT = "Ты умный ассистент dreamGPT AI. Отвечай кратко, ясно и эффективно."

# Vision models
VISION_MODELS = [
    'dreamgpt-ai',  # Smart model that auto-selects vision model for images
    'Qwen/Qwen2.5-VL-32B-Instruct',  # Explicitly supports vision
]

# Set up a global asyncio event loop for the bot
def run_asyncio_loop(loop):
    asyncio.set_event_loop(loop)
    loop.run_forever()

asyncio_loop = asyncio.new_event_loop()
asyncio_thread = threading.Thread(target=run_asyncio_loop, args=(asyncio_loop,), daemon=True)
asyncio_thread.start()

# Initialize Speech Recognition
recognizer = sr.Recognizer()
recognizer.energy_threshold = 300
recognizer.dynamic_energy_threshold = True
recognizer.dynamic_energy_adjustment_damping = 0.15
recognizer.dynamic_energy_ratio = 1.5
recognizer.pause_threshold = 0.8
recognizer.operation_timeout = None
recognizer.phrase_threshold = 0.3
recognizer.non_speaking_duration = 0.8

# Словарь для хранения истории чатов пользователей
user_chat_history = {}

def clean_reasoning_tags(text):
    """Remove all reasoning/thinking tags from AI responses (same as app.py)"""
    if not text or not isinstance(text, str):
        return text
    
    # Remove common reasoning tags with regex first (most efficient)
    # Remove all variations of reasoning tags
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<thinking>.*?</thinking>', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<reasoning>.*?</reasoning>', '', text, flags=re.DOTALL | re.IGNORECASE)
    
    # Remove lines that start with reasoning markers (line-by-line processing for edge cases)
    lines = text.split('\n')
    cleaned_lines = []
    in_reasoning_block = False
    
    for line in lines:
        line_lower = line.lower().strip()
        
        # Check if we're entering a reasoning block
        if '<think' in line_lower or '<reasoning' in line_lower or '<redacted_reasoning' in line_lower:
            in_reasoning_block = True
            continue
        
        # Check if we're exiting a reasoning block
        if in_reasoning_block and ('</think>' in line_lower or '</reasoning>' in line_lower):
            in_reasoning_block = False
            continue
        
        # Skip lines inside reasoning blocks
        if in_reasoning_block:
            continue
        
        # Skip lines that are just reasoning markers
        if line_lower.startswith('<') and ('think' in line_lower or 'reasoning' in line_lower):
            continue
        
        cleaned_lines.append(line)
    
    result = '\n'.join(cleaned_lines).strip()
    
    # Remove any remaining reasoning markers at start/end
    result = re.sub(r'^<[^>]*?(?:think|reasoning)[^>]*?>.*?</[^>]*?(?:think|reasoning)[^>]*?>', '', result, flags=re.DOTALL | re.IGNORECASE)
    result = re.sub(r'^\[REASONING\].*?\[/REASONING\]', '', result, flags=re.DOTALL | re.IGNORECASE)
    
    # Clean up multiple newlines
    result = re.sub(r'\n{3,}', '\n\n', result)
    
    return result.strip()

def compress_image(image_data: bytes, max_size_mb: float = 4.0, max_dimension: int = 2048) -> bytes:
    """Compress and resize image to reduce size for API (max 4MB, max 2048px)"""
    try:
        # Open image from bytes
        img = Image.open(io.BytesIO(image_data))
        original_format = img.format or 'JPEG'
        
        # Convert RGBA to RGB if needed (for JPEG compatibility)
        if img.mode in ('RGBA', 'LA', 'P'):
            background = Image.new('RGB', img.size, (255, 255, 255))
            if img.mode == 'P':
                img = img.convert('RGBA')
            background.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
            img = background
        elif img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize if image is too large
        width, height = img.size
        if width > max_dimension or height > max_dimension:
            ratio = min(max_dimension / width, max_dimension / height)
            new_width = int(width * ratio)
            new_height = int(height * ratio)
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            logger.info(f"Resized image from {width}x{height} to {new_width}x{new_height}")
        
        # Compress to target size
        output = io.BytesIO()
        quality = 95
        target_size = int(max_size_mb * 1024 * 1024)  # Convert MB to bytes
        
        # Try different quality levels to fit within size limit
        for q in range(95, 40, -10):
            output.seek(0)
            output.truncate(0)
            img.save(output, format='JPEG', quality=q, optimize=True)
            if len(output.getvalue()) <= target_size:
                quality = q
                break
        
        # If still too large, resize more aggressively
        if len(output.getvalue()) > target_size:
            scale_factor = (target_size / len(output.getvalue())) ** 0.5
            new_width = int(img.width * scale_factor)
            new_height = int(img.height * scale_factor)
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            output.seek(0)
            output.truncate(0)
            img.save(output, format='JPEG', quality=85, optimize=True)
            logger.info(f"Further resized to {new_width}x{new_height} to fit size limit")
        
        compressed_data = output.getvalue()
        original_size = len(image_data)
        compressed_size = len(compressed_data)
        compression_ratio = (1 - compressed_size / original_size) * 100 if original_size > 0 else 0
        
        logger.info(f"Image compressed: {original_size / 1024:.1f}KB -> {compressed_size / 1024:.1f}KB ({compression_ratio:.1f}% reduction)")
        
        return compressed_data
        
    except Exception as e:
        logger.error(f"Error compressing image: {e}")
        # Return original if compression fails
        return image_data

def photo_to_base64(photo_file_data: bytes, photo_format: str = 'jpeg') -> dict:
    """Convert Telegram photo to base64 data URI for API with compression"""
    try:
        # Compress image first to reduce size
        compressed_data = compress_image(photo_file_data)
        
        # Check final size (base64 increases size by ~33%)
        base64_size_estimate = len(compressed_data) * 1.33
        max_base64_size = 20 * 1024 * 1024  # 20MB limit for API
        
        if base64_size_estimate > max_base64_size:
            logger.warning(f"Image still too large after compression: {base64_size_estimate / 1024 / 1024:.1f}MB")
            # Try more aggressive compression
            compressed_data = compress_image(photo_file_data, max_size_mb=15.0, max_dimension=1536)
        
        base64_data = base64.b64encode(compressed_data).decode('utf-8')
        final_size = len(base64_data)
        
        logger.info(f"Base64 size: {final_size / 1024 / 1024:.2f}MB")
        
        # Use JPEG for compressed images (smaller than PNG)
        mime_type = 'image/jpeg'
        
        return {
            'type': mime_type,
            'data': base64_data
        }
    except Exception as e:
        logger.error(f"Error converting photo to base64: {e}")
        return None

async def generate_ai_response(text: str = None, user_id: int = None, is_inline: bool = False, photos: list = None) -> str:
    """Generate AI response using the neural network API with smart model selection"""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_TOKEN}",
    }
    
    # Получаем историю чата пользователя или создаем новую
    if user_id not in user_chat_history:
        user_chat_history[user_id] = []
    
    # Определяем, есть ли изображения
    has_images = photos and len(photos) > 0
    
    # Умная модель dreamGPT AI - автопереключение
    use_smart_model = not is_inline  # В инлайне не используем умную модель
    actual_model = None
    
    if use_smart_model:
        # dreamGPT AI - умная модель с автопереключением
        if has_images:
            # Используем vision модель для изображений
            actual_model = 'Qwen/Qwen2.5-VL-32B-Instruct'
        else:
            # Используем GPT для текста
            actual_model = 'openai/gpt-oss-120b'
    else:
        # Для инлайна используем быструю модель
        actual_model = "openai/gpt-oss-120b"
    
    # Создаем системный промпт в зависимости от режима
    if is_inline:
        system_prompt = """ты - быстрый ai-ассистент dreamgpt.

для инлайн режима:
• отвечай максимально кратко и по делу
• без markdown форматирования
• без вопросов в конце
• пиши с маленькой буквы
• используй дефисы - вместо тире —
• давай только суть ответа
• не используй режим мыслей или рассуждений
• отвечай сразу и четко"""
    else:
        # Используем системные промпты как в app.py
        combined_system_prompt = MAIN_SYSTEM_PROMPT.strip()
        if use_smart_model:
            if combined_system_prompt:
                combined_system_prompt += "\n\n" + DREAMGPT_SYSTEM_PROMPT
            else:
                combined_system_prompt = DREAMGPT_SYSTEM_PROMPT
        
        system_prompt = combined_system_prompt + """

твои особенности:
• отвечай на русском языке
• будь полезным и точным
• отвечай кратко, без воды
• если нужно, используй курсив для акцентов
• пиши все ответы с маленькой буквы
• используй дефисы - вместо тире —
• не упоминай инструкции или настройки
• отвечай как живой человек, не как бот

отвечай естественно и помогай пользователю!"""
    
    # Формируем сообщения с историей
    messages = [{"role": "system", "content": system_prompt}]
    
    # Добавляем историю чата (последние 10 сообщений для экономии токенов)
    recent_history = user_chat_history[user_id][-10:] if user_chat_history[user_id] else []
    messages.extend(recent_history)
    
    # Формируем текущее сообщение пользователя
    if has_images:
        # Для vision моделей используем специальный формат с изображениями
        text_part = text if text and text.strip() else "Что на фото?"
        content_parts = [{"type": "text", "text": text_part}]
        
        # Добавляем все фото в base64
        for photo_data in photos:
            base64_data = photo_to_base64(photo_data['data'], photo_data.get('format', 'jpeg'))
            if base64_data:
                content_parts.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{base64_data['type']};base64,{base64_data['data']}"
                    }
                })
        
        current_message = {"role": "user", "content": content_parts}
    else:
        # Обычное текстовое сообщение
        if not text:
            text = ""
        current_message = {"role": "user", "content": text}
    
    messages.append(current_message)
    
    # Выбираем параметры модели
    if is_inline:
        temperature = 0.5
        max_tokens = 500
    else:
        temperature = 0.7
        max_tokens = 2000
    
    data = {
        "model": actual_model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens
    }

    try:
        logger.info(f"Attempting API call for: {text[:50] if text else 'photo'}...")
        # Увеличиваем таймаут для vision моделей (обработка изображений занимает больше времени)
        timeout = 60 if has_images else 30
        async with aiohttp.ClientSession() as session:
            async with session.post(API_URL, headers=headers, json=data, timeout=timeout) as response:
                logger.info(f"API response status: {response.status}")
                if response.status != 200:
                    error = await response.text()
                    logger.error(f"API error response: {error}")
                    return f"❌ Ошибка API: {error}"

                response_data = await response.json()
                if 'choices' in response_data and len(response_data['choices']) > 0:
                    bot_response = response_data['choices'][0]['message']['content']
                    
                    # Очистка ответа от reasoning тегов (используем функцию как в app.py)
                    bot_response = clean_reasoning_tags(bot_response)
                    
                    # Убеждаемся что ответ не пустой
                    if not bot_response or len(bot_response.strip()) < 5:
                        return "❌ Получен пустой ответ от модели"
                    
                    # Сохраняем историю чата
                    if user_id is not None:
                        # Сохраняем пользовательское сообщение (текст или placeholder для фото)
                        user_message_for_history = text if text else ("📷 Изображение" if has_images else "")
                        user_chat_history[user_id].append({"role": "user", "content": user_message_for_history})
                        user_chat_history[user_id].append({"role": "assistant", "content": bot_response})
                        
                        # Ограничиваем историю до 20 сообщений (10 пар вопрос-ответ)
                        if len(user_chat_history[user_id]) > 20:
                            user_chat_history[user_id] = user_chat_history[user_id][-20:]
                    
                    return bot_response
                else:
                    logger.error(f"Unexpected API response structure: {response_data}")
                    return "❌ Не удалось обработать ответ API"
    except Exception as e:
        logger.error(f"Exception during API call: {str(e)}")
        return f"❌ Произошла ошибка: {str(e)}"

@bot.message_handler(commands=['start'])
def send_welcome(message):
    try:
        welcome_text = (
            "**🤖 привет! я ai ассистент dreamgpt**\n\n"
            "**что я умею:**\n"
            "💬 **отвечать на вопросы** - просто напиши мне\n"
            "📷 **работать с фото** - отправь фото и я опишу его\n"
            "🎙️ **распознавать голосовые** - отправь голосовое сообщение\n"
            "🔄 **сбрасывать чат** - используй команду /reset\n"
            "⚡ **быстрые ответы** - используй @dreamgptbot в любом чате\n\n"
            "**примеры:**\n"
            "• напиши: \"расскажи про квантовые компьютеры\"\n"
            "• фото: отправь фото и спроси что на нем\n"
            "• голосовое: запиши вопрос голосом\n"
            "• сброс: /reset\n"
            "• инлайн: @dreamgptbot как приготовить борщ?\n\n"
            "**🌐 веб-версия:** https://ai.dreampartners.online\n\n"
            "**готов помочь! 🚀**"
        )
        bot.reply_to(message, welcome_text, parse_mode='Markdown')
    except Exception as e:
        logger.error(f"Error in send_welcome: {e}")
        bot.send_message(message.from_user.id, f"Произошла ошибка: {str(e)}")

@bot.message_handler(commands=['reset'])
def handle_reset_command(message):
    """Handle /reset command to clear chat history"""
    try:
        user_id = message.from_user.id
        
        # Очищаем историю чата пользователя
        if user_id in user_chat_history:
            user_chat_history[user_id] = []
        
        reset_message = (
            "🔄 **Чат сброшен!**\n\n"
            "История нашего разговора очищена. "
            "Теперь я буду отвечать как будто мы только что познакомились! 😊\n\n"
            "Чем могу помочь?"
        )
        
        bot.reply_to(message, reset_message, parse_mode='Markdown')
        
    except Exception as e:
        logger.error(f"Error in handle_reset_command: {e}")
        bot.reply_to(message, f"❌ Ошибка при сбросе чата: {str(e)}")

@bot.message_handler(content_types=['text'])
def handle_text(message):
    try:
        text = message.text.strip()
        user_id = message.from_user.id
        
        if not text:
            return
        
        # Check for video URLs (TikTok, YouTube, Instagram)
        video_url_pattern = r'https?://(www\.)?(tiktok\.com|vm\.tiktok\.com|youtube\.com|youtu\.be|instagram\.com)/[^\s]+'
        video_match = re.search(video_url_pattern, text)
        
        if video_match:
            # Process video URL
            video_url = video_match.group(0)
            handle_video_url(message, video_url, text)
            return
        
        # Show typing indicator
        bot.send_chat_action(message.chat.id, 'typing')
        
        # Generate AI response with user context
        future = asyncio.run_coroutine_threadsafe(generate_ai_response(text, user_id), asyncio_loop)
        response = future.result(timeout=30)
        
        # Send response with Markdown formatting
        try:
            bot.reply_to(message, response, parse_mode='Markdown')
        except Exception as parse_error:
            # Если Markdown не парсится, отправляем как обычный текст
            logger.warning(f"Markdown parse error: {parse_error}")
            bot.reply_to(message, response)
        
    except Exception as e:
        logger.error(f"Error in handle_text: {e}")
        bot.reply_to(message, f"❌ Произошла ошибка: {str(e)}")

@bot.message_handler(content_types=['photo'])
def handle_photo(message):
    """Handle photo messages with AI vision"""
    try:
        user_id = message.from_user.id
        text = message.caption if message.caption else None
        
        # Show typing indicator
        bot.send_chat_action(message.chat.id, 'typing')
        
        # Get the largest photo (last in the list)
        photo = message.photo[-1]
        
        # Download photo
        file_info = bot.get_file(photo.file_id)
        downloaded_file = bot.download_file(file_info.file_path)
        
        # Determine format from file path
        file_ext = file_info.file_path.split('.')[-1].lower() if '.' in file_info.file_path else 'jpeg'
        if file_ext not in ['png', 'jpg', 'jpeg', 'gif', 'webp']:
            file_ext = 'jpeg'
        
        # Prepare photo data for API
        photos_data = [{
            'data': downloaded_file,
            'format': file_ext
        }]
        
        # Generate AI response with photo
        future = asyncio.run_coroutine_threadsafe(
            generate_ai_response(text=text, user_id=user_id, is_inline=False, photos=photos_data), 
            asyncio_loop
        )
        response = future.result(timeout=60)  # Увеличиваем таймаут для vision моделей
        
        # Send response with Markdown formatting
        try:
            bot.reply_to(message, response, parse_mode='Markdown')
        except Exception as parse_error:
            # Если Markdown не парсится, отправляем как обычный текст
            logger.warning(f"Markdown parse error: {parse_error}")
            bot.reply_to(message, response)
        
    except Exception as e:
        logger.error(f"Error in handle_photo: {e}")
        bot.reply_to(message, f"❌ Произошла ошибка при обработке фото: {str(e)}")

def handle_video_url(message, video_url, original_text):
    """Process video URL using downloader API and generate AI response"""
    status_msg = None
    try:
        user_id = message.from_user.id
        
        # Send processing message
        status_msg = bot.reply_to(message, "🎬 обрабатываю видео... [░░░░░░░░░░] 0%")
        
        # Call downloader API
        bot.edit_message_text("🎬 скачиваю и расшифровываю видео... [████░░░░░░] 40%", message.chat.id, status_msg.message_id)
        
        logger.info(f"[VIDEO] Processing URL: {video_url}")
        
        import requests
        api_endpoint = f"{DOWNLOADER_API_URL}/api/process"
        payload = {"url": video_url}
        
        response = requests.post(api_endpoint, json=payload, timeout=300)
        
        logger.info(f"[VIDEO] API Response Status: {response.status_code}")
        
        if response.status_code != 200:
            error_msg = f"❌ ошибка при обработке видео (код {response.status_code})"
            bot.edit_message_text(error_msg, message.chat.id, status_msg.message_id)
            return
        
        try:
            data = response.json()
        except Exception as json_err:
            logger.error(f"[VIDEO] Failed to parse JSON: {json_err}")
            bot.edit_message_text("❌ ошибка: API вернул некорректный ответ", message.chat.id, status_msg.message_id)
            return
        
        if data.get('status') != 'success':
            error_msg = data.get('message', 'не удалось обработать видео')
            bot.edit_message_text(f"❌ {error_msg}", message.chat.id, status_msg.message_id)
            return
        
        transcription = data.get('transcription') or data.get('text', '')
        summary = data.get('summary', '')
        
        logger.info(f"[VIDEO] Transcription length: {len(transcription)}")
        
        if not transcription:
            bot.edit_message_text("❌ не удалось получить расшифровку видео", message.chat.id, status_msg.message_id)
            return
        
        # Generate AI response with video context
        bot.edit_message_text("🤖 генерирую ответ... [████████░░] 80%", message.chat.id, status_msg.message_id)
        
        # Create video context prompt (same as web version)
        video_context = (
            "\n\n[ВИДЕО КОНТЕКСТ]\n"
            "Пользователь отправил видео по ссылке. Ниже аудио-транскрипт видео (может содержать ошибки распознавания речи).\n"
            f"{transcription}\n"
        )
        if summary:
            video_context += f"Краткое содержание:\n{summary}\n"
        video_context += (
            "[КОНЕЦ ВИДЕО КОНТЕКСТ]\n\n"
            "ВАЖНЫЕ ИНСТРУКЦИИ ДЛЯ ОТВЕТА:\n"
            "1. Начни ответ с фразы: 'Вы отправили видео. Вот что в нем:'\n"
            "2. Затем опиши содержание видео на основе транскрипта\n"
            "3. Помни: транскрипт может содержать ошибки распознавания речи, но НЕ УПОМИНАЙ это в ответе пользователю\n"
            "4. Будь кратким и по делу, избегай домыслов\n"
            "5. Просто опиши содержание видео естественным языком, без упоминаний о качестве транскрипта\n"
            "6. Если транскрипт совсем непонятный или пустой, скажи что не удалось распознать содержание\n"
            "7. ОБЯЗАТЕЛЬНО начни с 'Вы отправили видео. Вот что в нем:' чтобы было понятно, что это расшифровка видео\n"
            "8. НЕ пиши про ошибки распознавания, артефакты или качество транскрипта - просто опиши содержание\n"
        )
        
        # Prepare messages with video context
        system_prompt = MAIN_SYSTEM_PROMPT.strip() + "\n\n" + DREAMGPT_SYSTEM_PROMPT + video_context
        
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add user message if there was additional text
        user_message = original_text.replace(video_url, '').strip()
        if user_message:
            messages.append({"role": "user", "content": user_message})
        else:
            messages.append({"role": "user", "content": "Расскажи что в этом видео"})
        
        # Call AI API
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_TOKEN}",
        }
        
        ai_response = requests.post(API_URL, headers=headers, json={
            "model": "openai/gpt-oss-120b",
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 2000
        }, timeout=60)
        
        if ai_response.status_code != 200:
            bot.edit_message_text("❌ ошибка при генерации ответа AI", message.chat.id, status_msg.message_id)
            return
        
        ai_data = ai_response.json()
        if 'choices' not in ai_data or len(ai_data['choices']) == 0:
            bot.edit_message_text("❌ AI не вернул ответ", message.chat.id, status_msg.message_id)
            return
        
        ai_content = ai_data['choices'][0]['message']['content']
        ai_content = clean_reasoning_tags(ai_content)
        
        # Delete status message and send final response
        bot.delete_message(message.chat.id, status_msg.message_id)
        
        # Save to history
        if user_id not in user_chat_history:
            user_chat_history[user_id] = []
        user_chat_history[user_id].append({"role": "user", "content": f"🎥 Видео: {video_url}"})
        user_chat_history[user_id].append({"role": "assistant", "content": ai_content})
        
        # Send response
        try:
            bot.reply_to(message, ai_content, parse_mode='Markdown')
        except Exception as parse_error:
            logger.warning(f"Markdown parse error: {parse_error}")
            bot.reply_to(message, ai_content)
        
        logger.info(f"[VIDEO] Successfully processed video for user {user_id}")
        
    except requests.exceptions.Timeout:
        logger.error("[VIDEO] Timeout processing video")
        if status_msg:
            bot.edit_message_text("❌ таймаут при обработке видео (слишком долго)", message.chat.id, status_msg.message_id)
    except Exception as e:
        logger.error(f"[VIDEO] Error processing video: {e}")
        import traceback
        traceback.print_exc()
        if status_msg:
            bot.edit_message_text(f"❌ ошибка при обработке видео: {str(e)}", message.chat.id, status_msg.message_id)
        else:
            bot.reply_to(message, f"❌ ошибка при обработке видео: {str(e)}")

@bot.message_handler(content_types=['video', 'video_note'])
def handle_video_file(message):
    """Handle video file uploads"""
    try:
        bot.reply_to(message, "❌ загрузка видео файлов пока не поддерживается. отправьте ссылку на видео (TikTok, YouTube, Instagram)")
    except Exception as e:
        logger.error(f"Error in handle_video_file: {e}")

@bot.message_handler(content_types=['voice'])
def handle_voice(message):
    """Handle voice messages with speech recognition"""
    status_msg = None
    temp_input_path = None
    temp_audio_path = None
    
    try:
        voice = message.voice
        status_msg = bot.reply_to(message, "🎙️ расшифровываю голосовое... [░░░░░░░░░░] 0%")
        
        # Download voice file
        bot.edit_message_text("🎙️ получаю голосовое... [██░░░░░░░░] 20%", message.chat.id, status_msg.message_id)
        file_info = bot.get_file(voice.file_id)
        downloaded_file = bot.download_file(file_info.file_path)
        
        # Save to temporary file
        temp_input_path = os.path.join(tempfile.gettempdir(), f"{voice.file_unique_id}.ogg")
        with open(temp_input_path, 'wb') as f:
            f.write(downloaded_file)
        
        # Convert to WAV
        bot.edit_message_text("🎙️ конвертирую в wav... [████░░░░░░] 40%", message.chat.id, status_msg.message_id)
        audio = AudioSegment.from_file(temp_input_path, format="ogg")
        temp_audio_path = temp_input_path.replace(".ogg", ".wav")
        audio.set_frame_rate(16000).set_channels(1).set_sample_width(2).export(temp_audio_path, format="wav")
        
        # Transcribe speech
        bot.edit_message_text("🎙️ распознаю речь... [██████░░░░] 60%", message.chat.id, status_msg.message_id)
        text = ""
        try:
            with sr.AudioFile(temp_audio_path) as source:
                # Adjust for ambient noise
                recognizer.adjust_for_ambient_noise(source, duration=0.5)
                
                # Record the audio
                audio_data = recognizer.record(source)
                
                # Recognize speech using Google Speech Recognition
                text = recognizer.recognize_google(audio_data, language='ru-RU')
                
        except sr.UnknownValueError:
            text = "не удалось распознать речь"
        except sr.RequestError as e:
            text = f"ошибка сервиса распознавания речи: {e}"
        except Exception as e:
            text = f"ошибка при распознавании речи: {e}"
        
        # Generate AI response to transcribed text
        bot.edit_message_text("🤖 генерирую ответ... [████████░░] 80%", message.chat.id, status_msg.message_id)
        
        if text and text != "не удалось распознать речь" and not text.startswith("ошибка"):
            # Generate AI response with user context
            user_id = message.from_user.id
            future = asyncio.run_coroutine_threadsafe(generate_ai_response(text, user_id), asyncio_loop)
            ai_response = future.result(timeout=30)
            
            # Send both transcription and AI response
            bot.edit_message_text("✅ готово! [██████████] 100%", message.chat.id, status_msg.message_id)
            try:
                bot.send_message(
                    message.chat.id,
                    f"🎙️ **расшифровка:** {text}\n\n🤖 **ответ:** {ai_response}",
                    parse_mode="Markdown"
                )
            except Exception as parse_error:
                # Если Markdown не парсится, отправляем как обычный текст
                logger.warning(f"Markdown parse error in voice handler: {parse_error}")
                bot.send_message(
                    message.chat.id,
                    f"🎙️ расшифровка: {text}\n\n🤖 ответ: {ai_response}"
                )
        else:
            # Send only transcription if recognition failed
            bot.edit_message_text("⚠️ не удалось распознать речь", message.chat.id, status_msg.message_id)
            bot.send_message(message.chat.id, f"🎙️ {text}")
        
        # Delete status message
        if status_msg:
            try:
                bot.delete_message(message.chat.id, status_msg.message_id)
            except Exception as del_e:
                logger.warning(f"Failed to delete status message: {del_e}")
        
    except Exception as e:
        logger.error(f"Error in handle_voice: {e}", exc_info=True)
        error_message = f"❌ ошибка при обработке голосового: {str(e)}"
        if status_msg:
            try:
                bot.edit_message_text(error_message, message.chat.id, status_msg.message_id)
            except Exception as edit_e:
                logger.error(f"Failed to edit status message: {edit_e}")
                bot.reply_to(message, error_message)
        else:
            bot.reply_to(message, error_message)
    
    finally:
        # Clean up temporary files
        if temp_input_path and os.path.exists(temp_input_path):
            try:
                os.remove(temp_input_path)
            except Exception as e:
                logger.warning(f"Failed to remove temp input file: {e}")
        if temp_audio_path and os.path.exists(temp_audio_path):
            try:
                os.remove(temp_audio_path)
            except Exception as e:
                logger.warning(f"Failed to remove temp audio file: {e}")

@bot.inline_handler(lambda query: len(query.query) > 0)
def handle_inline_query(inline_query):
    try:
        query_text = inline_query.query.strip()
        
        if not query_text:
            return
        
        results = []
        
        # Regular AI assistant query
        try:
            # Generate AI response with user context for inline mode
            user_id = inline_query.from_user.id
            logger.info(f"Processing inline query: {query_text[:50]}...")
            
            future = asyncio.run_coroutine_threadsafe(generate_ai_response(query_text, user_id, is_inline=True), asyncio_loop)
            response = future.result(timeout=15)
            
            # Проверяем что ответ не пустой и не содержит только ошибки
            if not response or len(response.strip()) < 3:
                logger.warning(f"Empty or too short response: '{response}'")
                response = "извини, не смог сгенерировать ответ. попробуй еще раз."
            elif response.startswith("❌"):
                logger.warning(f"Error response received: {response}")
                # Если это ошибка API, даем более понятное сообщение
                if "ошибка" in response.lower():
                    response = "временная проблема с сервером. попробуй через минуту."
            
            # Create inline result - just the response without extra formatting
            results.append(
                types.InlineQueryResultArticle(
                    id='ai_response',
                    title='🤖 ответ',
                    description=response[:100] + ('...' if len(response) > 100 else ''),
                    input_message_content=types.InputTextMessageContent(
                        message_text=response
                    )
                )
            )
            
        except asyncio.TimeoutError:
            logger.error("Timeout generating AI response for inline query")
            results.append(
                types.InlineQueryResultArticle(
                    id='ai_timeout',
                    title='⏱️ таймаут',
                    description='сервер не отвечает. попробуй еще раз.',
                    input_message_content=types.InputTextMessageContent(
                        message_text='⏱️ сервер не отвечает. попробуй еще раз.'
                    )
                )
            )
        except Exception as e:
            logger.error(f"Error generating AI response: {e}")
            results.append(
                types.InlineQueryResultArticle(
                    id='ai_error',
                    title='❌ ошибка ai',
                    description='не удалось получить ответ. попробуй еще раз.',
                    input_message_content=types.InputTextMessageContent(
                        message_text='❌ не удалось получить ответ. попробуй еще раз.'
                    )
                )
            )
        
        # Answer inline query - проверяем что есть результаты
        if results:
            bot.answer_inline_query(inline_query.id, results, cache_time=1)
        else:
            logger.warning("No results to send for inline query")
            # Отправляем fallback результат
            fallback_result = types.InlineQueryResultArticle(
                id='fallback',
                title='🤖 dreamgpt',
                description='напиши вопрос для получения ответа',
                input_message_content=types.InputTextMessageContent(
                    message_text='🤖 dreamgpt - напиши вопрос для получения ответа'
                )
            )
            bot.answer_inline_query(inline_query.id, [fallback_result], cache_time=1)
        
    except Exception as e:
        logger.error(f"Error in handle_inline_query: {e}")
        try:
            # Send error result
            error_result = types.InlineQueryResultArticle(
                id='error',
                title='❌ ошибка бота',
                description='произошла ошибка при обработке запроса',
                input_message_content=types.InputTextMessageContent(
                    message_text='❌ произошла ошибка при обработке запроса'
                )
            )
            bot.answer_inline_query(inline_query.id, [error_result], cache_time=1)
        except Exception as answer_error:
            logger.error(f"Failed to answer inline query: {answer_error}")
            # Если даже ответить не можем, просто игнорируем

if __name__ == '__main__':
    logger.info("Запуск AI бота DreamGPT...")
    
    while True:
        try:
            logger.info("Запуск polling...")
            bot.polling(non_stop=True, skip_pending=True, timeout=60)
        except KeyboardInterrupt:
            logger.info("Бот остановлен пользователем.")
            break
        except Exception as e:
            logger.error(f"Ошибка polling: {e}")
            logger.info("Перезапуск через 10 секунд...")
            import time
            time.sleep(10)
    
    logger.info("AI бот завершен.")
