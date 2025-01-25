CKPT_PATH=$1 python -m uvicorn server.spinUpServer:app --reload &
streamlit run ./gui/mainGui.py 
