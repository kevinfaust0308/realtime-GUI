1. Clone repo and install dependencies in requirements.txt
2. Run app.py to launch GUI
3. Set the capture window by clicking where the top-left corner should be and where the
bottom-right corner should be


### Extending new models

There's 4 things to add for including a new model.
1. Update the dropdown categories (at top of app.py)
2. Create a metadata file for knowing where to load the model from, the classes to display on gui, etc (There is a field I have mpp=0.504 which basically means the model was trained on 20x magnification but I am not using it since all models are trained on 20x and the user should be viewing the slide on 20x)
3. Updating load_model in utils.py
4. Creating a file called process_region_X.py which returns the frame (or an annotated frame to display) + text to write on the GUI