#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;
int main(int argc,char** argv)
{
     VideoCapture cap(0); //capture the video from webcam
     if ( !cap.isOpened() )  // if not success, exit program
        {
            cout << "Cannot open the web cam" << endl;
            return -1;
        }

     //HARDCODING THE SKIN TONE COLORS
     int iLowH = 0;
     int iHighH = 20;
     int iLowS = 88;
     int iHighS = 199;
     int iLowV = 66;
     int iHighV = 255;
     //Create trackbars in "Results" window
        namedWindow("Results", CV_WINDOW_AUTOSIZE); //create a window called "Control"
        createTrackbar("LowH", "Results", &iLowH, 179); //Hue (0 - 179)
        createTrackbar("HighH", "Results", &iHighH, 179);
        createTrackbar("LowS", "Results", &iLowS, 255); //Saturation (0 - 255)
        createTrackbar("HighS", "Results", &iHighS, 255);
        createTrackbar("LowV", "Results", &iLowV, 255);//Value (0 - 255)
        createTrackbar("HighV", "Results", &iHighV, 255);

        int iMinBlobSize = 1;
        createTrackbar("MinBlobSize*1000", "Results", &iMinBlobSize, 100);

    while (true)
    {
        Mat handsqaureOriginal;
               bool bSuccess = cap.read(handsqaureOriginal);
               if (!bSuccess) //if not success, break loop
               {
                   cout << "Cannot read a frame from video stream" << endl;
                   break;
               }


Mat handHSV;

//CONVERT THE IMAGE TO HSV

cvtColor( handsqaureOriginal,handHSV,COLOR_BGR2HSV);

Mat handThresholded;

//CHECKING THE MINIMUM &MAXIMUM COLOR RANGE OF IMAGE

inRange(handHSV,Scalar(iLowH,iLowS,iLowV),Scalar(iHighH,iHighS,iHighV),handThresholded);

//DISPLAY THRESHOLDED IMAGE
imshow("Thresholded Image",handThresholded);

//MORPHOLOGICAL OPENING & CLOSING WILL REMOVE THE SMALL OBJECTS FROM FOREGROUND
//OPENING
erode(handThresholded,handThresholded,getStructuringElement(MORPH_ELLIPSE,Size(10,10)));
dilate(handThresholded,handThresholded,getStructuringElement(MORPH_ELLIPSE,Size(10,10)));

//CLOSING
dilate(handThresholded,handThresholded,getStructuringElement(MORPH_ELLIPSE,Size(10,10)));
erode(handThresholded,handThresholded,getStructuringElement(MORPH_ELLIPSE,Size(10,10)));


//blob analysis-----start

                vector<vector<Point>> contours; //TAKING POINTS FROM CONTOURS
                vector<Vec4i>hierarchy; //CREATING PARENT BLOB JUST CHECK THIS CONTOUR IS A PART OF PARENT CONTOUR
                findContours(handThresholded,contours,hierarchy,CV_RETR_TREE,CV_CHAIN_APPROX_SIMPLE,Point(0,0));

//ROTATED FINGER SQAURE

                       int n = contours.size();
                RotatedRect fingerRectRotated[10],smallfingerRect[10];
                for(int i = 0 ; i < n ; i++ )
                {
                    fingerRectRotated[i] = minAreaRect(contours[i]);
                    Point2f vertices[4],center1,center2;
                    // Point2f vertices[4],center2;
                     float angle1;
                             //Size2f size1; //gives width,height
                     double width1, height1;

                     angle1 =  fingerRectRotated[i].angle;
                            center1 = fingerRectRotated[i].center;
                            //fingerRectRotated[i].size(size1);
                            width1 = fingerRectRotated[i].size.width;
                            height1 = fingerRectRotated[i].size.height;
                             fingerRectRotated[i].points(vertices);

                             if (fingerRectRotated[i].size.width > fingerRectRotated[i].size.height)
                                     {
                                    center2.x =  (int)round(center1.x + width1 *1/3* cos(angle1 * CV_PI / 180.0));
                                    center2.y =  (int)round(center1.y + width1*1/3* sin(angle1 * CV_PI / 180.0));


                                    smallfingerRect[i] = RotatedRect(center2, Size2f(1.5*height1,height1), angle1);
                                     }
                                     else
                                     {
                                         angle1 = 180 - fingerRectRotated[i].angle;
                                         center2.y =  (int)round(center1.y + height1 *1/3* cos(angle1 * CV_PI / 180.0));
                                         center2.x =  (int)round(center1.x + height1 *1/3* sin(angle1 * CV_PI / 180.0));


                                         smallfingerRect[i] = RotatedRect(center2, Size2f(width1,1.3*width1), 180-angle1);
                                     }
                             smallfingerRect[i].points(vertices);

                    fingerRectRotated[i].points(vertices);

                    for(int j = 0 ; j < 4 ; j++ )
                    {
                        line(handsqaureOriginal,vertices[j],vertices[(j+1)%4],Scalar(0,255,0),1,8);
                    }

                }
                // 2. Get the moments
                      vector<Moments> mu(contours.size() );
                      for( int i = 0; i < contours.size(); i++ )
                      {
                          mu[i] = moments( contours[i], false );
                      }

                     //  3. Get the mass centers (centroids)
                      //
                      vector<Point2f> mc( contours.size() );
                      for( int i = 0; i < contours.size(); i++ )
                      {
                          mc[i] = Point2f( mu[i].m10/mu[i].m00 , mu[i].m01/mu[i].m00 );
                      }

                      // 4. Draw blobs
                             RNG rng(12345);
                             Mat blobImg = Mat::zeros( handThresholded.size(), CV_8UC3 );
                             for( int i = 0; i< contours.size(); i++ )
                             {
                                 Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
                                 drawContours( blobImg, contours, i, color, 2, 8, hierarchy, 0, Point() );
                             }


                             // 5. Blob filtering
                              //
                              vector<vector<Point>> contours_filt;
                              vector<Vec4i> hierarchy_filt;
                              vector<Moments> mu_filt;
                              vector<Point2f> mc_filt;

                            // Filter blobs w.r.t. to size
                              int minBlobArea = iMinBlobSize*1000; // Minimum Pixels for valid blob
                              for( int i = 0; i< contours.size(); i++ )
                              {
                                  if (mu[i].m00 > minBlobArea)
                                  {
                                      contours_filt.push_back(contours[i]);
                                      hierarchy_filt.push_back(hierarchy[i]);
                                      mu_filt.push_back(mu[i]);
                                      mc_filt.push_back(mc[i]);
                                  }
                              }

                              // 6. Draw filtered blobs
                                    //
                                    Mat blobImg_filt = Mat::zeros(handThresholded.size(), CV_8UC3 );;
                                    for( int i = 0; i< contours_filt.size(); i++ )
                                    {
                                        Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
                                        drawContours( blobImg_filt, contours_filt, i, color, 2, 8, hierarchy_filt, 0, Point() );
                                        circle( handsqaureOriginal, mc_filt[i], 4, color, -1, 8, 0 );
                                    }
                imshow("final Result",handsqaureOriginal);

                if (waitKey(30) == 27) //wait for 'esc' key
                {
                    cout << "esc key is pressed by user" << endl;
                    break;
                }
    }
                return 0;

}
