# Get Python 3.11 Debian Bookworm image
FROM python:3.11-bookworm

# Install pip requirements
COPY ../requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# Run bash
CMD ["/bin/bash"]
