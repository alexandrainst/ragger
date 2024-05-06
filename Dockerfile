FROM python:3.11-slim-bookworm

# Set up the demo port
ENV DEMO_PORT=7860
EXPOSE $DEMO_PORT

# Install poetry
RUN pip install "poetry==1.8.2"

# Set up the user
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
	PATH=/home/user/.local/bin:$PATH

# Move the files into the container
WORKDIR /project
COPY --chown=user . /project

# Install dependencies
RUN poetry env use python3.11
RUN poetry install --no-interaction --no-cache --without dev

# Run the script
CMD poetry run python src/scripts/run_demo.py demo.host=0.0.0.0 demo.port=$DEMO_PORT
