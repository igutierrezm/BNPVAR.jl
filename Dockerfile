# Install R 4.2.1
FROM rocker/verse:4.2.1
RUN apt-get update

# Install julia 1.7.3
WORKDIR /opt/
ARG JULIA_TAR=julia-1.7.3-linux-x86_64.tar.gz
RUN wget -nv https://julialang-s3.julialang.org/bin/linux/x64/1.7/${JULIA_TAR}
RUN tar -xzf ${JULIA_TAR}
RUN rm -rf ${JULIA_TAR}
RUN ln -s /opt/julia-1.7.3/bin/julia /usr/local/bin/julia

# Make R visible to RCall
RUN echo export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:`R RHOME`/lib" >> ~/.bashrc

### Instructions ###

# To create an image from this Dockerfile, run (in terminal)
    # sudo docker build -t bnpvar:0.1 .
# To create a container using this Docker image, run (in terminal)
    # sudo docker run <docker/rocker options> bnpvar:0.1
# e.g.
    # docker run \
    #   --rm -d \
    #   -p 8787:8787 \
    #   -e "ROOT=TRUE" \
    #   -e USER=rstudio \
    #   -e PASSWORD=123 \
    #   -v $HOME/.gitconfig:/home/rstudio/.gitconfig \
    #   -v $HOME/.ssh:/home/rstudio/.ssh \
    #   bnpvar:0.1
# Visit https://www.rocker-project.org/ for more details.