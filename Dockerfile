from centos
RUN yum install python3 -y
RUN pip3 install matplotlib numpy --user
RUN python3 -m pip --no-cache-dir install --upgrade \
    pip \
    setuptools
RUN pip3 install sklearn --user
RUN pip3 install tensorflow --user
RUN pip3 install keras --user
ENTRYPOINT ["python"]
CMD ["./cnn.py"]
