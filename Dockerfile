FROM python:3.6

RUN mkdir -p /usr/src/app
WORKDIR /usr/src/app

ADD . /usr/src/app/

# http://danielnouri.org/notes/2012/12/19/libblas-and-liblapack-issues-and-speed,-with-scipy-and-ubuntu/
RUN apt-get update
RUN apt-get install -y libatlas3-base

RUN pip install -r requirements.txt

ENTRYPOINT [ "gunicorn", "web_project.wsgi", "-b 0.0.0.0:8000" ]
