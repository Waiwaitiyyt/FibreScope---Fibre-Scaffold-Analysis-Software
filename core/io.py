import os
from tkinter import filedialog
import json

def selectImg():
    filetypes = [
        ("Image Files", "*.png *.jpg *.jpeg *.bmp *.gif *.tiff *.tif"),
        ("PNG Files", "*.png"),
        ("JPEG Files", "*.jpg *.jpeg")]

    filename = filedialog.askopenfilename(
    title='Open a file',
    initialdir='E:\CoraMetix\Fibre Diameter Measurement\Scaffold Analyser\scaffoldAnalysis_Dev\validate\testSet',
    filetypes=filetypes)
    with open("data.json", "r") as jsonFile:
        data = json.load(jsonFile)
    data["imgPath"] = filename
    with open("data.json", "w") as jsonFile:
        json.dump(data, jsonFile, indent = 2)
    

def saveResultImg():
    pass

def changeFibreModel():
    filetypes = [
        ("Model Files", "*.pth")]
    filename = filedialog.askopenfilename(
    title='Open a file',
    initialdir='/',
    filetypes=filetypes)
    if filename != "":
        with open("data.json", "r") as jsonFile:
            data = json.load(jsonFile)
        data["fibreModel"] = filename
        with open("data.json", "w") as jsonFile:
            json.dump(data, jsonFile, indent = 2)

def changePoreModel():
    filetypes = [
        ("Model Files", "*.pth")]
    filename = filedialog.askopenfilename(
    title='Open a file',
    initialdir='/',
    filetypes=filetypes)
    if filename != "":
        with open("data.json", "r") as jsonFile:
            data = json.load(jsonFile)
        data["poreModel"] = filename
        with open("data.json", "w") as jsonFile:
            json.dump(data, jsonFile, indent = 2)

def changeScaleFactor():
    pass


if __name__ == "__main__":
    pass