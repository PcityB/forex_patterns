from lightning.app import LightningWork, LightningApp, LightningFlow
import subprocess

class FlaskServer(LightningWork):
    def run(self):
        subprocess.Popen(["python", "main.py"])

class RootFlow(LightningFlow):
    def __init__(self):
        super().__init__()
        self.server = FlaskServer()

    def run(self):
        self.server.run()

app = LightningApp(RootFlow())
