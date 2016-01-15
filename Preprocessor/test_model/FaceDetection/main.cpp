#include "model.h"
#include "params.h"
#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <boost/program_options.hpp>
#include <boost/filesystem/path.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/info_parser.hpp>
#include <algorithm>

using namespace std;
namespace po = boost::program_options;
using boost::filesystem::path;
using boost::property_tree::ptree;



int main(int argc, char *argv[])
{	
	path ModelFile;
	path faceDetectorFilename;
	path faceBoxesDirectory;
	path outputDirectory;
	path inputPaths;
	path tainPaths;

	try {
		po::options_description desc("Allowed options");
		desc.add_options()
			("help,h",
			"Produce help message")			
			("input,i", po::value<path>(&inputPaths)->required(),
			"The path of image.")
			("model,m", po::value<path>(&ModelFile)->required(),
			"An  model file to load.")
			("train,t", po::value<path>(&tainPaths)->required(),
			"The train.txt.")
			("face-detector,f", po::value<path>(&faceDetectorFilename)->required(),
			"Path to an XML CascadeClassifier from OpenCV.")
			;

		po::positional_options_description p;
		p.add("input", -1);

		po::variables_map vm;
		po::store(po::command_line_parser(argc, argv).options(desc).positional(p).run(), vm);
		if (vm.count("help")) {
			std::cout << "Usage: detect-landmarks [options]" << std::endl;
			std::cout << desc;
			return EXIT_SUCCESS;
		}
		po::notify(vm);
		if (vm.count("face-detector")  != 1) {
			std::cout << "Error while parsing command-line arguments: specify either a face-detector (-f)  as input" << std::endl;
			std::cout << desc;
			return EXIT_SUCCESS;
		}

	}
	catch (po::error& e) {
		std::cout << "Error while parsing command-line arguments: " << e.what() << std::endl;
		std::cout << "Use --help to display a list of options." << std::endl;
		return EXIT_SUCCESS;
	}

	ptree pt;
	sParams params;

	try {
		read_info(tainPaths.string(), pt);
	}
	catch (const boost::property_tree::ptree_error& e) {
		std::cout << std::string("Error reading the config file: ") + e.what() << std::endl;
		return -EXIT_FAILURE;
	}

	//init mean face
	ptree meanface = pt.get_child("meanface");	
	params.__facebox_scale_factor = meanface.get<double>("SCALE_FACTOR");
	params.__facebox_scale_const = meanface.get<double>("SCALE_CONST");
	params.__facebox_width_div = meanface.get<double>("WIDTH_DIV");
	params.__facebox_height_div = meanface.get<double>("HEIGHT_DIV");

	cv::CascadeClassifier faceCascade;

	if (!faceCascade.load(faceDetectorFilename.string()))
	{
		std::cout << "Error loading the face detection model." << std::endl;
		return EXIT_FAILURE;
	}
  string filePath = inputPaths.string();
  cv::Mat img = cv::imread(filePath, 0);
	if (img.empty()){
		std::cout << "Error loading the image." << std::endl;
		return EXIT_FAILURE;
	}
  cv::Mat img_dis = cv::imread(filePath);
  cv::Mat img_patch = cv::imread(filePath);
  int fileNameRecounter = filePath.length();
  int pointIndex = -1;
  bool pointFlag = false;
  string fileNameSb = "";
  string filePathSb = "";
  for (int i = fileNameRecounter - 1; i >= 0; i--)
  {
    if (filePath[i] != '.' && pointFlag != true) {
      continue;
    }
    else if (pointFlag != true) {
      pointFlag = true;
      pointIndex = i;
      continue;
    }
    if (filePath[i] == '/' || filePath[i] == '\\')
    {
      fileNameSb += filePath.substr(i + 1, pointIndex - 1 - i);
      filePathSb += filePath.substr(0, i + 1);
      break;
    }
  }

  if (fileNameSb.length() > 3 && fileNameSb[0] == 'p' && fileNameSb[1] == 'a' && fileNameSb[2] == 't')
  {
    return 0;
  }

	std::vector<cv::Rect> detectedFaces;
	faceCascade.detectMultiScale(img, detectedFaces, 1.2, 2, 0, cv::Size(50, 50));

	if (!detectedFaces.empty()){
		cModel app(ModelFile.string(), &params);
		app.Init();
		sModel model = app.GetModel();
		int stages = model.__head.__num_stage;
		cv::Mat_<float> meanface = model.__meanface;		

    for (int i = 0; i < detectedFaces.size(); i++) {
      cv::Mat_<float> shape = app.Reshape_alt(meanface, detectedFaces[i]);
      double t = (double)cvGetTickCount();
      for (int j = 0; j < stages; j++){
        cv::Mat_<int> binary = app.DerivBinaryfeat(img, detectedFaces[i], shape, j);
        app.UpDate(binary, detectedFaces[i], shape, j);
      }
      t = (double)cvGetTickCount() - t;
      //std::cout << "Alignment runtime:" << t / (cvGetTickFrequency() * 1000) << " ms" << std::endl;

      if (shape.rows < 68)
      {
        return 0;
      }

      int patchCounter = 0;
      int patchOrgLen = 90;
      int scaleDelta = 24;
      int blockHeight = (std::min(img_patch.rows, (int)shape(55, 1) + patchOrgLen) - std::max(0, (int)shape(38, 1) - patchOrgLen)) / 2;
      int blockDelta = blockHeight / 3;

      // ´¦Àí´Ö¶ÔÆëºóµÄÐ¡¿é
      for (int blk = 0; blk < 5; blk++) {
        for (int sc = 0; sc < 3; sc++) {
          // È¡Õû¸öroi
          if (blk == 4)
          {
            int patchLen = patchOrgLen - scaleDelta * sc;
            int global_cut_startx = std::max(0, (int)shape(38, 0) - patchLen); // ×óÑÛx
            int global_cut_starty = std::max(0, (int)shape(38, 1) - patchLen); // ×óÑÛy
            int global_cut_endx = std::min(img_patch.cols, (int)shape(55, 0) + patchLen); // ÓÒ×ì½Çx
            int global_cut_endy = std::min(img_patch.rows, (int)shape(55, 1) + patchLen); // ÓÒ×ì½Çy
            cv::Range rop_x;
            rop_x.start = global_cut_startx;
            rop_x.end = global_cut_endx;
            cv::Range rop_y;
            rop_y.start = global_cut_starty;
            rop_y.end = global_cut_endy;
            cv::Mat masker = cv::Mat(img_patch, rop_y, rop_x);
            //cv::imshow("result - origin", masker);
            //cv::waitKey();
            char nameBuffer[64];
            itoa(patchCounter++, nameBuffer, 10);
            resize(masker, masker, cvSize(31, 39));
            imwrite(filePathSb + "pat_" + fileNameSb + "_" + string(nameBuffer) + ".jpg", masker);
            // Gray it
            cv::Mat ograyer;
            cvtColor(masker, ograyer, CV_BGR2GRAY);
            cv::Mat grayer;
            cvtColor(ograyer, grayer, CV_GRAY2RGB);
            //cv::imshow("result - gray", grayer);
            //cv::waitKey();
            itoa(patchCounter++, nameBuffer, 10);
            resize(grayer, grayer, cvSize(31, 39));
            imwrite(filePathSb + "pat_" + fileNameSb + "_" + string(nameBuffer) + ".jpg", grayer);
          }
          else
          {
            int patchLen = patchOrgLen - scaleDelta * sc;
            int global_cut_startx = std::max(0, (int)shape(38, 0) - patchLen); // ×óÑÛx
            int global_cut_endx = std::min(img_patch.cols, (int)shape(55, 0) + patchLen); // ÓÒ×ì½Çx
            int global_cut_starty = std::max(0, (int)shape(38, 1) - patchOrgLen + blockDelta * blk); // ×óÑÛy
            int global_cut_endy = std::min(img_patch.rows, global_cut_starty + blockHeight); // ÓÒ×ì½Çy
            cv::Range rop_x;
            rop_x.start = global_cut_startx;
            rop_x.end = global_cut_endx;
            cv::Range rop_y;
            rop_y.start = global_cut_starty;
            rop_y.end = global_cut_endy;
            cv::Mat masker = cv::Mat(img_patch, rop_y, rop_x);
            //cv::imshow("result - origin", masker);
            //cv::waitKey();
            char nameBuffer[64];
            itoa(patchCounter++, nameBuffer, 10);
            resize(masker, masker, cvSize(31, 39));
            imwrite(filePathSb + "pat_" + fileNameSb + "_" + string(nameBuffer) + ".jpg", masker);
            // Gray it
            cv::Mat ograyer;
            cvtColor(masker, ograyer, CV_BGR2GRAY);
            cv::Mat grayer;
            cvtColor(ograyer, grayer, CV_GRAY2RGB);
            //cv::imshow("result - gray", grayer);
            //cv::waitKey();
            itoa(patchCounter++, nameBuffer, 10);
            resize(grayer, grayer, cvSize(31, 39));
            imwrite(filePathSb + "pat_" + fileNameSb + "_" + string(nameBuffer) + ".jpg", grayer);
          }

        }
      }

      patchOrgLen = 70;
      scaleDelta = 16;
      // ´¦Àí¾Ö²¿Ð¡¿é
      const int ar[5] = {38, 45, 34, 49, 55};

			for (int mf = 0; mf < 5; mf++) {
        int m = ar[mf];
				//cv::circle(img_dis, cv::Point((int)shape(m, 0), (int)shape(m, 1)), 1, cv::Scalar(0, 255, 0));
        // ·½ÐÎpatch£º×óÑÛ¡¢ÓÒÑÛ¡¢±Ç¡¢×ó×ì½Ç¡¢ÓÒ×ì½Ç
        //if (m == 38 || m == 45 || m == 34 || m == 49 || m == 55) {
          int or_y = (int)shape(m, 1);
          int or_x = (int)shape(m, 0);
          for (int sc = 0; sc < 3; sc++) {
            int patchLen = patchOrgLen - scaleDelta * sc;
            int cut_x = std::max(0, or_x - patchLen);
            int cut_y = std::max(0, or_y - patchLen);
            cv::Range rop_x;
            rop_x.start = cut_x;
            rop_x.end = std::min(img_patch.cols, or_x + patchLen);
            cv::Range rop_y;
            rop_y.start = cut_y;
            rop_y.end = std::min(img_patch.rows, or_y + patchLen);
            //std::cout << cut_x << "," << cut_y << "," << rop_x.end << "," << rop_y.end << std::endl;
            cv::Mat masker = cv::Mat(img_patch, rop_y, rop_x);
            //cv::imshow("result - origin", masker);
            //cv::waitKey();
            char nameBuffer[64];
            itoa(patchCounter++, nameBuffer, 10);
            resize(masker, masker, cvSize(31, 31));
            imwrite(filePathSb + "pat_" + fileNameSb + "_" + string(nameBuffer) + ".jpg", masker);
            // Gray
            cv::Mat ograyer;
            cvtColor(masker, ograyer, CV_BGR2GRAY);
            cv::Mat grayer;
            cvtColor(ograyer, grayer, CV_GRAY2RGB);
            //cv::imshow("result - gray", flipper);
            //cv::waitKey();
            itoa(patchCounter++, nameBuffer, 10);
            resize(grayer, grayer, cvSize(31, 31));
            imwrite(filePathSb + "pat_" + fileNameSb + "_" + string(nameBuffer) + ".jpg", grayer);
          }
        //}
			}
		}


    std::cout << "success: " << filePathSb << fileNameSb << std::endl;
		//cv::imshow("result", img_dis);
		//cv::waitKey();
	}else{
		std::cout << "No faces detect!." << std::endl;
	}

	return 0;
}