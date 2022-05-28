import gradio as gr
from mentalhealth import MentalHealth

def launchGUI():
    MN = MentalHealth()
    iface = gr.Interface(fn=MN._runMentalHealthAlarmSystem, inputs="text", outputs="dataframe")
    iface.launch(server_name="0.0.0.0", server_port=8080, share=True)

if __name__ == "__main__":
    launchGUI()