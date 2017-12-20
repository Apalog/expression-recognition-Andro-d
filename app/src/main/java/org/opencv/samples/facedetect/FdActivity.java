package org.opencv.samples.facedetect;

import android.drm.DrmStore;
import android.os.CountDownTimer;
import android.app.Activity;
import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.opengl.GLSurfaceView;
import android.os.Bundle;
import android.os.Handler;
import android.util.Log;
import android.view.MenuItem;
import android.view.TextureView;
import android.view.View;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.TextView;
import android.widget.Toast;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2;
import org.opencv.android.CameraRenderer;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Range;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;

public class FdActivity extends Activity implements CvCameraViewListener2 {

    private static final String TAG = "OCVSample::Activity";
    private static final Scalar FACE_RECT_COLOR = new Scalar(0, 255, 0, 255);
    private static final Scalar FACE_CIRCLE_COLOR = new Scalar(0, 255, 0, 255);
    private static final Scalar FACE_EMOTION = new Scalar(255, 255, 255, 255);
    public static final int JAVA_DETECTOR = 0;
    public static final int NATIVE_DETECTOR = 1;

    private Mat mRgba;
    private Mat mGray;
    private File mCascadeFile;
    private CascadeClassifier mJavaDetector;
    private DetectionBasedTracker mNativeDetector;
    private Rect[] facesArray;

    private int mDetectorType = JAVA_DETECTOR;
    private String[] mDetectorName;

    private float mRelativeFaceSize = 0.2f;
    private int mAbsoluteFaceSize = 0;

    private static final String MODEL_FILE = "file:///android_asset/constant_graph_weights.pb";
    private static final String MODEL_FILE_INCEPTION = "file:///android_asset/inception_resnet.pb";

    private TensorFlowInferenceInterface inferenceInterface;
    private TensorFlowInferenceInterface inferenceInterface_inception;
    private Bitmap mInputBitmap;
    private Bitmap mInputBitmap_inception;
    private Bitmap imageMean;
    private Bitmap imageStd;
    private static Bitmap imageMean_inception;
    private static Bitmap imageStd_inception;
    private Mat imgMeanMat;
    private Mat imgStdMat;
    private Mat imgMeanMat_inception;
    private Mat imgStdMat_inception;
    private String inputName = "conv2d_1_input";
    private int inputSize = 40;
    private String outputName = "output_node0";
    private String inputName_inception = "input_1";
    private int inputSize_inception = 49;
    private String outputName_inception = "output_node_inception0";
    private static String[] emotions = {"NEUTRAL", "HAPPINESS", "SAD", "SURPRISE", "ANGER"};
    private String emotion_face;

    private float play;
    float[] player1 = new float[5];

    private boolean emotionClick = false;
    private boolean faceClick = false;
    private boolean playClick = false;
    private String[] choicePlay = {"HAPPINESS","ANGER", "SURPRISE", "SAD"};

    private CameraBridgeViewBase mOpenCvCameraView;

    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS: {
                    Log.i(TAG, "OpenCV loaded successfully");

                    // Load native library after(!) OpenCV initialization
                    System.loadLibrary("detection_based_tracker");
                    try {
                        System.loadLibrary("tensorflow_inference");
                    } catch (UnsatisfiedLinkError e2) {
                        throw new RuntimeException("Native TF methods not found; check that the correct native libraries are present and loaded.");
                    }

                    // Load img mean & std
                    imageMean = BitmapFactory.decodeResource(getResources(), R.drawable.mean_image);
                    imageStd = BitmapFactory.decodeResource(getResources(), R.drawable.std_image);
                    imageMean_inception = BitmapFactory.decodeResource(getResources(), R.drawable.in_mean_image_49);
                    imageStd_inception = BitmapFactory.decodeResource(getResources(), R.drawable.in_std_image_49);

                    // convert to Mat before resize
                    imgMeanMat = convertBitmapToMat(imageMean);
                    imgStdMat = convertBitmapToMat(imageStd);
                    imgMeanMat_inception = convertBitmapToMat(imageMean_inception);
                    imgStdMat_inception = convertBitmapToMat(imageStd_inception);

//                    Log.i("image channel :", String.valueOf(imgMeanMat_inception.channels()));


                    Imgproc.resize(imgMeanMat, imgMeanMat, new Size(40, 40));
                    Imgproc.resize(imgStdMat, imgStdMat, new Size(40, 40));
                    Imgproc.resize(imgMeanMat_inception, imgMeanMat_inception, new Size(inputSize_inception, inputSize_inception));
                    Imgproc.resize(imgStdMat_inception, imgStdMat_inception, new Size(inputSize_inception, inputSize_inception));

                    // convert Mat to Bitmap before treatment
                    imageStd = convertMatToBitmap(imgStdMat);
                    imageMean = convertMatToBitmap(imgMeanMat);
                    imageStd_inception = convertMatToBitmap(imgStdMat_inception);
                    imageMean_inception = convertMatToBitmap(imgMeanMat_inception);

                    try {
                        // load cascade file from application resources
                        InputStream is = getResources().openRawResource(R.raw.lbpcascade_frontalface);
                        File cascadeDir = getDir("cascade", Context.MODE_PRIVATE);
                        mCascadeFile = new File(cascadeDir, "lbpcascade_frontalface.xml");
                        FileOutputStream os = new FileOutputStream(mCascadeFile);

                        byte[] buffer = new byte[4096];
                        int bytesRead;
                        while ((bytesRead = is.read(buffer)) != -1) {
                            os.write(buffer, 0, bytesRead);
                        }
                        is.close();
                        os.close();

                        mJavaDetector = new CascadeClassifier(mCascadeFile.getAbsolutePath());
                        if (mJavaDetector.empty()) {
                            Log.e(TAG, "Failed to load cascade classifier");
                            mJavaDetector = null;
                        } else
                            Log.i(TAG, "Loaded cascade classifier from " + mCascadeFile.getAbsolutePath());

                        mNativeDetector = new DetectionBasedTracker(mCascadeFile.getAbsolutePath(), 0);

                        cascadeDir.delete();

                    } catch (IOException e) {
                        e.printStackTrace();
                        Log.e(TAG, "Failed to load cascade. Exception thrown: " + e);
                    }
                    mOpenCvCameraView.setCameraIndex(1);
                    mOpenCvCameraView.enableView();
                }
                break;
                default: {
                    super.onManagerConnected(status);
                }
                break;
            }
        }

    };

    public FdActivity() {
        mDetectorName = new String[2];
        mDetectorName[JAVA_DETECTOR] = "Java";
        mDetectorName[NATIVE_DETECTOR] = "Native (tracking)";

        Log.i(TAG, "Instantiated new " + this.getClass());
    }

    /**
     * Called when the activity is first created.
     */
    @Override
    public void onCreate(Bundle savedInstanceState) {
        Button b1, b2, b3, b4;
        Log.i(TAG, "called onCreate");
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        setContentView(R.layout.face_detect_surface_view);

        setUpInferenceInterface();
        b1 = (Button) findViewById(R.id.button_emotion);
        b1.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if (!emotionClick) {
                    emotionClick = true;
                } else {
                    emotionClick = false;
                }
            }
        });
        b2 = (Button) findViewById(R.id.button_detect);
        b2.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if (!faceClick) {
                    faceClick = true;
                } else {
                    faceClick = false;
                }
            }
        });
        b4 = (Button) findViewById(R.id.button_play);
        b4.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if (!playClick) {
                    playClick= true;
                } else {
                    playClick = false;
                }
            }
        });


        mOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.fd_activity_surface_view);
        mOpenCvCameraView.setVisibility(CameraBridgeViewBase.VISIBLE);
        mOpenCvCameraView.setCvCameraViewListener(this);
    }

    @Override
    public void onPause() {
        super.onPause();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    @Override
    public void onResume() {
        super.onResume();
        if (!OpenCVLoader.initDebug()) {
            Log.d(TAG, "Internal OpenCV library not found. Using OpenCV Manager for initialization");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_0_0, this, mLoaderCallback);
        } else {
            Log.d(TAG, "OpenCV library found inside package. Using it!");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
    }

    public void onDestroy() {
        super.onDestroy();
        mOpenCvCameraView.disableView();
    }

    public void onCameraViewStarted(int width, int height) {
        mGray = new Mat();
        mRgba = new Mat();
    }

    public void onCameraViewStopped() {
        mGray.release();
        mRgba.release();
    }

    public Mat onCameraFrame(CvCameraViewFrame inputFrame) throws InterruptedException {
        mRgba = inputFrame.rgba();
        mGray = inputFrame.gray();
        if (mAbsoluteFaceSize == 0) {
            int height = mGray.rows();
            if (Math.round(height * mRelativeFaceSize) > 0) {
                mAbsoluteFaceSize = Math.round(height * mRelativeFaceSize);
            }
            mNativeDetector.setMinFaceSize(mAbsoluteFaceSize);
            mOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.fd_activity_surface_view);

        }
        MatOfRect faces = new MatOfRect();
        if (mDetectorType == JAVA_DETECTOR) {
            if (mJavaDetector != null)
                mJavaDetector.detectMultiScale(mGray, faces, 1.1, 2, 2, // TODO: objdetect.CV_HAAR_SCALE_IMAGE
                        new Size(mAbsoluteFaceSize, mAbsoluteFaceSize), new Size());
        } else if (mDetectorType == NATIVE_DETECTOR) {
            if (mNativeDetector != null)
                mNativeDetector.detect(mGray, faces);
        } else {
            Log.e(TAG, "Detection method is not selected!");
        }
        facesArray = faces.toArray();
        float[] stock = new float[facesArray.length];
        if(facesArray.length != 0) {
            for (int i = 0; i < facesArray.length; i++) {
                // Growing factor
                int f = (int) (facesArray[i].width * 0.2);
                if (emotionClick) {
                    Rect rect = new Rect(facesArray[i].x, facesArray[i].y, facesArray[i].width, facesArray[i].height);
                    // Check emotion,
                    Mat subMats_inception = mRgba.submat(rect);
                    //Mat subMats_inception = mGray.submat(rect);
                    Log.i("SUBMAT :",String.valueOf(subMats_inception.type()));
                    Imgproc.resize(subMats_inception, subMats_inception, new Size(inputSize_inception, inputSize_inception));
                    mInputBitmap_inception = convertMatToBitmap(subMats_inception);
                    emotion_face = witchOne(mInputBitmap_inception);
                    stock[0] = play;
                    switch (emotion_face) {
                        case "ANGER":
                            //red
                            Imgproc.putText(mRgba, emotion_face, new Point(rect.x, rect.y), 2, 2, new Scalar(255, 0, 0, 1));
                            break;
                        case "HAPPINESS":
                            //yellow
                            Imgproc.putText(mRgba, emotion_face, new Point(rect.x, rect.y), 2, 2, new Scalar(255, 255, 0, 1));
                            break;
                        case "SAD":
                            //grey
                            Imgproc.putText(mRgba, emotion_face, new Point(rect.x, rect.y), 2, 2, new Scalar(255, 255, 255, 1));
                            break;
                        case "SURPRISE":
                            //blue
                            Imgproc.putText(mRgba, emotion_face, new Point(rect.x, rect.y), 2, 2, new Scalar(0, 0, 255, 1));
                            break;
                    }
                }
                //check value for Feature point recognition
                if (faceClick) {
                    // Print rect
                    Imgproc.rectangle(mRgba, new Point(facesArray[i].x - f, facesArray[i].y - f), new Point((facesArray[i].x + facesArray[i].width) + f, (facesArray[i].y + facesArray[i].height) + f), FACE_RECT_COLOR, 3);
                    // newRect -> Feature point recognition,  rect -> facial recognition of emotion
                    Rect newRect = new Rect(new Point(facesArray[i].x - f, facesArray[i].y - f), new Point((facesArray[i].x + facesArray[i].width) + f, (facesArray[i].y + facesArray[i].height) + f));

                    if (newRect.height > mRgba.height()) {
                        newRect.height = mRgba.height() - 1;
                    }
                    if (newRect.width > mRgba.width()) {
                        newRect.width = mRgba.width() - 1;
                    }
                    if (newRect.x <= 0) newRect.x = 0;
                    if (newRect.y <= 0) newRect.y = 0;
                    if (newRect.y + newRect.height >= mRgba.height()) {
                        newRect.y = mRgba.height() - newRect.height;
                    }
                    if (newRect.x + newRect.width >= mRgba.width()) {
                        newRect.x = mRgba.width() - newRect.width;
                    }

                    //After verification of it values
                    newRect = new Rect(newRect.x, newRect.y, newRect.width, newRect.height);
                    Mat subMats = mRgba.submat(newRect);
                    double x_P = newRect.tl().x;
                    double y_P = newRect.tl().y;
                    double y_D = subMats.cols();
                    double x_D = subMats.rows();
                    Imgproc.resize(subMats, subMats, new Size(40, 40));
                    mInputBitmap = convertMatToBitmap(subMats);
                    float[] outputsFace = face_analyse();
                    //Print points
                    for (int position = 0; position < outputsFace.length; position += 2) {
                        Imgproc.circle(mRgba, new Point((((outputsFace[position + 1] + 0.5)) * x_D) + x_P, (((outputsFace[position] + 0.5)) * y_D) + y_P), 5, FACE_CIRCLE_COLOR);
                    }
                }
            }
        }
        if(playClick == true && facesArray.length !=0){
            boolean great;
            while(order <= choicePlay.length+1 && playClick == true) {
                while (order ==4 && playClick == true) {
                    //Core.divide(mRgba, new Scalar(255, 255, 255, 1), mRgba);
                    Imgproc.putText(mRgba, " Score : " + String.valueOf((player1[0] * 100 + player1[1] * 100 + player1[2] * 100 + player1[3] * 100) / 4 + " %"), new Point(2, mRgba.cols() / 2), 2, 6, FACE_RECT_COLOR, 2);
                    return mRgba;
                }
                if (order != 4) {
                    great = onePlayer(choicePlay[order], order);
                    if (checkAll(great)) order++;
                }
                return mRgba;
            }
        }
        if(playClick == false){
            order = 0;
        }
        return mRgba;
    }
    int order = 0;

    // Verification of emotion
    private boolean checkAll(boolean great){
        if(great==false){
            return false;
        }
        return true;
    }

    private void showValues(Bitmap mInputBitmaps) {
        int color = mInputBitmaps.getPixel(20, 20);
        int A = (color >> 24) & 0xff; // or color >>> 24
        int R = (color >> 16) & 0xff;
        int G = (color >> 8) & 0xff;
        int B = (color) & 0xff;
    }

    private Bitmap convertMatToBitmap(Mat inputCrop) {
        Bitmap mInputBitmaps = Bitmap.createBitmap(inputCrop.cols(), inputCrop.rows(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(inputCrop, mInputBitmaps);
        return mInputBitmaps;
    }

    private Mat convertBitmapToMat(Bitmap inputBitmap) {
        // /!\ type : 24
        Mat img = new Mat(inputBitmap.getHeight(), inputBitmap.getWidth(), 24);
        Utils.bitmapToMat(inputBitmap, img);
        img.convertTo(img, 24);
        return img;
    }

    // Load models
    void setUpInferenceInterface() {
        inferenceInterface = new TensorFlowInferenceInterface(getAssets(), MODEL_FILE);
        inferenceInterface_inception = new TensorFlowInferenceInterface(getAssets(), MODEL_FILE_INCEPTION);
    }

    // Application model points feature
    public float[] face_analyse() {
        int[] intValues = new int[inputSize * inputSize];
        int[] intValuesStd = new int[inputSize * inputSize];
        int[] intValuesMean = new int[inputSize * inputSize];
        float[] floatValues = new float[inputSize * inputSize * 3];
        float[] outputsFaces = new float[136];
        mInputBitmap.getPixels(intValues, 0, mInputBitmap.getWidth(), 0, 0, mInputBitmap.getWidth(), mInputBitmap.getHeight());
        imageStd.getPixels(intValuesStd, 0, imageStd.getWidth(), 0, 0, imageStd.getWidth(), imageStd.getHeight());
        imageMean.getPixels(intValuesMean, 0, imageMean.getWidth(), 0, 0, imageMean.getWidth(), imageMean.getHeight());
        for (int i = 0; i < intValues.length; ++i) {
            final int val = intValues[i];
            final int valStd = intValuesStd[i];
            final int valMean = intValuesMean[i];
            floatValues[i * 3 + 0] = (((val >> 16) & 0xFF) - ((valMean >> 16) & 0xFF)) / ((valStd >> 16) & 0xFF);
            floatValues[i * 3 + 1] = (((val >> 8) & 0xFF) - ((valMean >> 8) & 0xFF)) / ((valStd >> 8) & 0xFF);
            floatValues[i * 3 + 2] = ((val & 0xFF) - (valMean & 0xFF)) / (valStd & 0xFF);
        }
        inferenceInterface.feed(inputName, floatValues, 1, inputSize, inputSize, 3);
        inferenceInterface.run(new String[]{outputName});
        inferenceInterface.fetch(outputName, outputsFaces);
        return outputsFaces;
    }

    // Application model emotions
    public float[] face_emotion(Bitmap mInputBitmap_inceptionE) {
        int[] intValues = new int[inputSize_inception * inputSize_inception];
        int[] intValuesStd = new int[inputSize_inception * inputSize_inception];
        int[] intValuesMean = new int[inputSize_inception * inputSize_inception];
        float[] floatValues = new float[inputSize_inception * inputSize_inception * 3];
        float[] outputsEmotion = new float[5];
        mInputBitmap_inceptionE.getPixels(intValues, 0, mInputBitmap_inceptionE.getWidth(), 0, 0, mInputBitmap_inceptionE.getWidth(), mInputBitmap_inceptionE.getHeight());
        imageStd_inception.getPixels(intValuesStd, 0, imageStd_inception.getWidth(), 0, 0, imageStd_inception.getWidth(), imageStd_inception.getHeight());
        imageMean_inception.getPixels(intValuesMean, 0, imageMean_inception.getWidth(), 0, 0, imageMean_inception.getWidth(), imageMean_inception.getHeight());
        for (int i = 0; i < intValues.length; ++i) {
            final int val = intValues[i];
            final int valStd = intValuesStd[i];
            final int valMean = intValuesMean[i];
            floatValues[i * 3 + 0] = (((val >> 16) & 0xFF) - ((valMean >> 16) & 0xFF)) / ((valStd >> 16) & 0xFF);
            floatValues[i * 3 + 1] = (((val >> 8) & 0xFF) - ((valMean >> 8) & 0xFF)) / ((valStd >> 8) & 0xFF);
            floatValues[i * 3 + 2] = ((val & 0xFF) - (valMean & 0xFF)) / (valStd & 0xFF);
        }
        inferenceInterface_inception.feed(inputName_inception, floatValues, 1, inputSize_inception, inputSize_inception,3);
        inferenceInterface_inception.run(new String[]{outputName_inception});
        inferenceInterface_inception.fetch(outputName_inception, outputsEmotion);
        return outputsEmotion;
    }

    // Select emotion
    public String witchOne(Bitmap mInputBitmap_inception) {
        int moreE = 0;
        float[] outputs = face_emotion(mInputBitmap_inception);
        for (int i = 1; i < outputs.length; i++) {
            if (outputs[moreE] < outputs[i]) {
                moreE = i;
            }
        }
        play = outputs[moreE];
        return emotions[moreE];
    }

    // Part to play
    private boolean onePlayer(String emotion, int nbEmotion) {
        Rect rect = new Rect(facesArray[0].x, facesArray[0].y, facesArray[0].width, facesArray[0].height);
        Mat subMats_inception = mRgba.submat(rect);
        Imgproc.resize(subMats_inception, subMats_inception, new Size(inputSize_inception, inputSize_inception));
        Bitmap mBitmapPlayer = convertMatToBitmap(subMats_inception);
        Imgproc.putText(mRgba, String.valueOf(emotion), new Point(2, mRgba.cols() / 2), 2, 6, FACE_RECT_COLOR, 2);
        if (witchOne(mBitmapPlayer) == emotion) {
            player1[nbEmotion] = play;
            return true;
        } else {
            return false;
        }

    }
}
