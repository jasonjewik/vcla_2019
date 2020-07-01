#!/bin/bash

# set arguments
ARGS[0]="null"
ARGS[1]="null"
IND=0
VERBOSE=0

# functions
print_usage_msg () {
    echo "Usage: ./get_frames.sh [-v VIDEO_FILE] [-o OUTPUT_FOLDER] [-V]"
    echo "-V means 'verbose'"
    exit $1
}

check_vid_file () {
    # check that the video file exists
    if [ ! -e $VIDEO_FILE ]
    then
        echo "Error with video file"
        echo "Could not open $VIDEO_FILE"
        exit 1
    fi
}

check_output_folder () {
    # clear output folder if it exists
    if [ -d $OUTPUT_FOLDER ]
    then
        rm -rf $OUTPUT_FOLDER
        mkdir $OUTPUT_FOLDER
    else
        mkdir $OUTPUT_FOLDER
    fi
}

# check for arguments
if [ $# -eq 0 ]
then
    echo "No arguments supplied"
    print_usage_msg 1
fi

# get arguments
while getopts :v:o:hV option
do case "${option}"
in
v) VIDEO_FILE=${OPTARG}; ARGS[IND]="video"; IND=`expr $IND + 1`;;
o) OUTPUT_FOLDER=${OPTARG}; ARGS[IND]="output"; IND=`expr $IND + 1`;;
h) print_usage_msg 0;;
V) VERBOSE=1;;
?) echo "Unrecognized argument"; print_usage_msg 1;;
esac
done

for i in "${ARGS[@]}"
do
    if [ $i = "video" ]
    then 
        check_vid_file
    elif [ $i = "output" ]
    then
        check_output_folder
    elif [ $i = "null" ]
    then
        echo "Missing argument"
        print_usage_msg 1
    fi
done

if [ $VERBOSE -eq 1 ]
then
    ffmpeg -i $VIDEO_FILE -vf fps=1 `readlink -f $OUTPUT_FOLDER`/out%d.png
    exit $?
else
    ffmpeg -i $VIDEO_FILE -vf fps=1 `readlink -f $OUTPUT_FOLDER`/out%d.png > /dev/null 2>&1
    STATUS=$?

    if [ $STATUS -eq 0 ]
    then
        echo "Success."
        exit 0
    else
        echo "Failed with exit code $STATUS"
        exit $STATUS
    fi
fi