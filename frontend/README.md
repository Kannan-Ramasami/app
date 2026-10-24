# SignSpeak Frontend

This is the frontend UI/UX for the SignSpeak project (Sign Language to Natural Text translation).
It is built entirely using Python and `customtkinter` to provide a modern, dark-themed, and responsive GUI that exactly matches the Figma design.

## Project Structure

```
frontend/
│
├── main.py                # Main application entry point & page router
├── README.md              # Frontend documentation (this file)
└── pages/                 # UI Pages for the application
    ├── __init__.py
    ├── home.py            # Main translation interface (Camera, Translated Text)
    ├── settings.py        # Settings page (404 Placeholder)
    ├── help.py            # Help page (404 Placeholder)
    └── not_found.py       # Generic 404 placeholder for unlinked buttons
```

## Setup & Requirements

The frontend relies on `customtkinter`. Ensure the environment has the required libraries installed:

```bash
pip install customtkinter pillow
```

Run the application:
```bash
cd frontend
python main.py
```

## Backend Integration Guide

Dear Backend Developer, 

Here is how you can connect your ML/translation backend to this frontend:

1. **Accessing the UI Elements**:
   The `HomePage` class in `frontend/pages/home.py` contains all the elements you need to update dynamically.
   
2. **Updating the Camera Feed**:
   Inside `home.py`, there is a method placeholder `update_camera_frame(self, frame_image)`. You can call this method from the backend to stream webcam frames to the UI. Ensure that the frames are converted to `customtkinter.CTkImage` format.
   
   *Example Usage:*
   ```python
   # Assuming 'app' is the main App instance
   home_page = app.pages["HomePage"]
   home_page.update_camera_frame(my_ctk_image)
   ```

3. **Updating the Translation Text**:
   You can push text to the "Translated Letters / Words" and "Translated Sentence" panels using the `update_translated_text(self, letters_words, sentence)` method in `home.py`.
   
   *Example Usage:*
   ```python
   home_page.update_translated_text("H E L L O", "Hello world!")
   ```

4. **Handling the START/STOP Button**:
   The `toggle_start` method in `home.py` is executed when the user clicks the START button. You can bind your backend logic (e.g., initiating camera capture, loading the ML model) inside this method. The `self.is_running` boolean tracks the state.

## Components Breakdown

- **Top Bar**: Contains the Logo, Contact button, Help (?), and Settings (⚙).
- **Camera Panel (Left)**: Designed to display the live feed. It's currently filled with a placeholder icon and "Enable Camera To Use".
- **Letters/Words Panel (Top Right)**: Displays real-time detected characters or isolated words.
- **Sentence Panel (Bottom Right)**: Displays the context-aware constructed sentence.
- **Bottom Bar**: Contains the START/STOP toggle, Language selection (🌐), and Record (◎) buttons.
