FROM python:3.11-slim-bookworm

# Set up the demo port
ENV DEMO_PORT=7860
EXPOSE $DEMO_PORT

# Install poetry
RUN pip install "poetry==1.8.2"

# Move the files into the container
WORKDIR /project
COPY . /project

# Install dependencies
RUN poetry env use python3.11
RUN poetry install --no-interaction --no-cache --without dev

# Run the script
CMD poetry run python src/scripts/run_demo.py host=0.0.0.0 port=$DEMO_PORT
