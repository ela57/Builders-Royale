FROM python:3.13-slim
RUN pip install --no-cache-dir numpy==2.1.3
WORKDIR /bot
COPY . /bot
RUN python3 /bot/precompute.py
ENTRYPOINT ["python3", "-u", "/bot/bot.py"]
