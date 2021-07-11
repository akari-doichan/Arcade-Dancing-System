# coding: utf-8
"""
####################
Command Line Scripts
####################

Run the program from a command line.

*************************************
1. Conversion from Video to Landmarks
*************************************

You can convert from `data/sample-instructor.mp4` to `data/sample-instructor_mediapipe_angle.json` by the following command:

.. code-block:: shell

    $ poetry run video2landmarks -V data/sample-instructor.mp4 \\
                                --model mediapipe \\
                                --score-method angle         -

*****************************************
2. Realtime dance with instructor's video
*****************************************

You can dance while scoring in real time using the data (`data/sample-instructor_mediapipe_angle.json`) created above.

.. code-block:: shell

    $ poetry run arcade-dance -J data/sample-instructor_mediapipe_angle.json \\
                              --connections body \\
                              --max-score 90 \\
                              --instructor-xywh "[-410,10,400,400]" \\
                              --codec MP4V \\
                              --record
"""