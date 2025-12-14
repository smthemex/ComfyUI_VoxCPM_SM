# 👩‍🍳 A Voice Chef's Guide

Welcome to the VoxCPM kitchen! Follow this recipe to cook up perfect generated speech. Let's begin.

---

## 🥚 Step 1: Prepare Your Base Ingredients (Content)

First, choose how you'd like to input your text:

### 1. Regular Text (Classic Mode)
- ✅ Keep "Text Normalization" ON. Type naturally (e.g., "Hello, world! 123"). The system will automatically process numbers, abbreviations, and punctuation using WeTextProcessing library.

### 2. Phoneme Input (Native Mode)
- ❌ Turn "Text Normalization" OFF. Enter phoneme text like `{HH AH0 L OW1}` (EN) or `{ni3}{hao3}` (ZH) for precise pronunciation control. In this mode, VoxCPM also supports native understanding of other complex non-normalized text—try it out!
- **Phoneme Conversion**: For Chinese, phonemes are converted using pinyin. For English, phonemes are converted using CMUDict. Please refer to the relevant documentation for more details.

---

## 🍳 Step 2: Choose Your Flavor Profile (Voice Style)

This is the secret sauce that gives your audio its unique sound.

### 1. Cooking with a Prompt Speech (Following a Famous Recipe)
- A prompt speech provides the desired acoustic characteristics for VoxCPM. The speaker's timbre, speaking style, and even the background sounds and ambiance will be replicated.
- **For a Clean, Denoising Voice:**
  - ✅ Enable "Prompt Speech Enhancement". This acts like a noise filter, removing background hiss and rumble to give you a pure, clean voice clone. However, this will limit the audio sampling rate to 16kHz, restricting the cloning quality ceiling.
- **For High-Quality Audio Cloning (Up to 44.1kHz):**
  - ❌ Disable "Prompt Speech Enhancement" to preserve all original audio information, including background atmosphere, and support audio cloning up to 44.1kHz sampling rate.

### 2. Cooking au Naturel (Letting the Model Improvise)
- If no reference is provided, VoxCPM becomes a creative chef! It will infer a fitting speaking style based on the text itself, thanks to the text-smartness of its foundation model, MiniCPM-4.
- **Pro Tip**: Challenge VoxCPM with any text—poetry, song lyrics, dramatic monologues—it may deliver some interesting results!

---

## 🧂 Step 3: The Final Seasoning (Fine-Tuning Your Results)

You're ready to serve! But for master chefs who want to tweak the flavor, here are two key spices.

### CFG Value (How Closely to Follow the Recipe)
- **Default**: A great starting point.
- **Voice sounds strained or weird?** Lower this value. It tells the model to be more relaxed and improvisational, great for expressive prompts.
- **Need maximum clarity and adherence to the text?** Raise it slightly to keep the model on a tighter leash.
- **Short sentences?** Consider increasing the CFG value for better clarity and adherence.
- **Long texts?** Consider lowering the CFG value to improve stability and naturalness over extended passages.

### Inference Timesteps (Simmering Time: Quality vs. Speed)
- **Need a quick snack?** Use a lower number. Perfect for fast drafts and experiments.
- **Cooking a gourmet meal?** Use a higher number. This lets the model "simmer" longer, refining the audio for superior detail and naturalness.

---

Happy creating! 🎉 Start with the default settings and tweak from there to suit your project. The kitchen is yours!

