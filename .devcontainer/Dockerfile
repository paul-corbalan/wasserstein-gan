# Get Python Debian Bookworm image
FROM python:bookworm

# Install pip requirements
COPY ../requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# Run bash
CMD ["/bin/bash"]
