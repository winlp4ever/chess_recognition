#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"



#include <iostream>

using namespace cv;
using namespace std;

typedef pair<Point, Point> p;

bool intersection(Point2f o1, Point2f p1, Point2f o2, Point2f p2,
	Point2f &r);

// return list of intersection points of hough lines found in chessboard image
vector<Point2f> hough(const char* filename, int nb_lines = 7) {
	// retrieve file and check if it exists
	Mat src = imread(filename, 0);
	assert(!src.empty() && "no such file!");

	Mat dst, cdst;
	Canny(src, dst, 50, 200, 3);
	cvtColor(dst, cdst, CV_GRAY2BGR);

	imwrite("../canny.jpg", dst);
	vector<Vec2f> lines;
	HoughLines(dst, lines, 1, CV_PI/180, 100, 0, 0);

	vector<p> red_lines, blue_lines;
	for( size_t i = 0; i < lines.size(); i++ ) {
		float rho = lines[i][0], theta = lines[i][1];
	
		Point pt1, pt2;
		double a = cos(theta), b = sin(theta);
		double x0 = a*rho, y0 = b*rho;
		pt1.x = cvRound(x0 + 1000 * (-b));
		pt1.y = cvRound(y0 + 1000 * (a));
		pt2.x = cvRound(x0 - 1000 * (-b));
		pt2.y = cvRound(y0 - 1000 * (a));
	
		p l(pt1, pt2);
		// check if it's a vertical line
		if ((theta < CV_PI / 180 * 20 || theta > CV_PI / 180 * 160) && red_lines.size() < nb_lines) {
			red_lines.push_back(l);
			line(cdst, pt1, pt2, Scalar(0, 0, 255), 1, CV_AA);
		}
		// check if it's a horizontal line
		if ((theta > CV_PI / 180 * 89 && theta < CV_PI / 180 * 91) && blue_lines.size() < nb_lines) {
			blue_lines.push_back(l);
			line(cdst, pt1, pt2, Scalar(0, 255, 255), 1, CV_AA);
		}
	}
	imwrite("../lines.jpg", cdst);

	// find intersection points
	vector<Point2f> list;
	for(vector<p>::iterator i_red = red_lines.begin(); i_red != red_lines.end(); ++i_red)
		for (vector<p>::iterator i_blue = blue_lines.begin(); i_blue != blue_lines.end(); ++i_blue) {
			Point pr = i_red->first,
				qr = i_red->second,
				pb = i_blue->first,
				qb = i_blue->second;
			Point2f X;
			if (intersection(pr, qr, pb, qb, X))
				list.push_back(X);
		}
	// draw intersection points
	int font = CV_FONT_HERSHEY_SCRIPT_SIMPLEX;
	for (int i = 0; i < list.size(); i++) {
		putText(cdst, to_string(i), list[i], font, 1, Scalar(168, 255, 249));
	}
	imshow("source", src);
	imshow("detected lines", cdst);

	waitKey();

	return list;
}

// Finds the intersection of two lines, or returns false.
// The lines are defined by (o1, p1) and (o2, p2).
bool intersection(Point2f o1, Point2f p1, Point2f o2, Point2f p2,
	Point2f &r)
{
	Point2f x = o2 - o1;
	Point2f d1 = p1 - o1;
	Point2f d2 = p2 - o2;

	float cross = d1.x*d2.y - d1.y*d2.x;
	if (abs(cross) < /*EPS*/1e-8)
		return false;

	double t1 = (x.x * d2.y - x.y * d2.x) / cross;
	r = o1 + d1 * t1;
	return true;
}



int main() {
	const char* filename = "../chessgame.png";
	vector<Point2f> L = hough(filename);
	
	cout << "nb of lines : " << L.size() << endl;
	cout << "nb of intersection points : " << L.size() << endl;
	Mat src = imread(filename);
	

	vector<Point2f> obj;
	vector<Point2f> scene;
	obj.push_back(L[6]);
	scene.push_back(Point(400, 200 + 100));
	obj.push_back(L[20]);
	scene.push_back(Point(700, 200 + 100));
	obj.push_back(L[0]);
	scene.push_back(Point(400, 200 + 600));
	obj.push_back(L[14]);
	scene.push_back(Point(700, 200 + 600));
	Mat H = findHomography(obj, scene);
	Mat K(1000, 800, CV_8U);
	warpPerspective(src, K, H, K.size());
	imshow("ret", K); waitKey(0);
	imwrite("../reconstruct.jpg", K);
	Mat inv = H.inv();
	cout << inv << endl;
	// return table in original image
	vector<Point2f> new_tab;
	vector<Point2f> old_tab;
	for (int i = 0; i < 9; i++)
		for (int j = 0; j < 9; j++)
			new_tab.push_back(Point(i * 100, 1000 - j * 100));
	perspectiveTransform(new_tab, old_tab, inv);
	cout << old_tab.size() << endl;


	Mat copy;
	src.copyTo(copy);
	for (vector<Point2f>::iterator it = old_tab.begin(); it != old_tab.end(); ++it) {
		drawMarker(copy, *it, Scalar(0, 0, 255), 1);
	}
	imshow("origin", copy); waitKey(0);
	imwrite("../res.jpg", copy);
	getchar();
	return 0;
}
