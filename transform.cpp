#include <iostream>
#include <fstream>
#include <filesystem>
#include <vector>
#include <optional>
#include <mutex>
#include <thread>
#include <deque>
#include <algorithm>
#include <cmath>
#include <cassert>

using namespace std;
namespace fs = std::filesystem;

#define unused(x) (void)x

static bool verbose = false;
const float angle_delta = 0.2;

struct directory_pair {
    fs::path source;
    fs::path target;
};

struct point {
    float x;
    float y;
    float z;
    float i;

    float angle();
};

inline float to_degrees(float radians) {
    return radians * (180.0 / M_PI);
}

float point::angle() {
    return 90 - to_degrees(acos(this->z / sqrt(pow(this->x, 2) + pow(this->y, 2) + pow(this->z, 2))));
}

const vector<directory_pair> point_dirs = {
    {"testing/velodyne_original", "testing/velodyne_filtered"},
    {"training/velodyne_original", "training/velodyne_filtered"},
};

const vector<directory_pair> label_dirs = {
    {"training/label_2", "training/label_2_filtered"},
};

struct angle_mapping {
    float angle;
    size_t ring64;
    size_t ring32;
    bool keep;
};

// angle mapping, sorted by angle (important).
const angle_mapping angle_mappings[64] = {
    {1.9367, 29,  22, true},
    {1.573966, 28,  17, true},
    {1.304757, 25,  13, true},
    {0.871566, 24,  18, true},
    {0.57880998, 3,  14, true},
    {0.180617, 2,  9, true},
    {-0.088762, 31,  5, true},
    {-0.45182899, 30,  10, true},
    {-0.80315, 27,  6, true},
    {-1.201239, 26,  1, true},
    {-1.49388, 21,  31, true},
    {-1.833245, 20,  2, true},
    {-2.207566, 17,  0, false},
    {-2.546633, 16,  27, true},
    {-2.8738379, 13,  23, true},
    {-3.235882, 12,  24, true},
    {-3.5393341, 23,  20, true},
    {-3.935853, 22,  19, true},
    {-4.2155242, 19,  15, true},
    {-4.5881028, 18,  16, true},
    {-4.9137921, 15,  0, false},
    {-5.2507782, 14,  12, true},
    {-5.6106009, 9,  0, false},
    {-5.958395, 8,  11, true},
    {-6.3288889, 5,  0, false},
    {-6.675746, 4,  0, false},
    {-6.9990368, 1,  0, false},
    {-7.287312, 0,  8, true},
    {-7.6787701, 11,  0, false},
    {-8.0580254, 10,  0, false},
    {-8.3104696, 7,  0, false},
    {-8.7114143, 6,  7, true},
    {-9.0260181, 61,  0, false},
    {-9.5735149, 60,  0, false},
    {-10.06249, 57,  0, false},
    {-10.470731, 56,  0, false},
    {-10.956945, 35,  0, false},
    {-11.598996, 34,  4, true},
    {-12.115005, 63,  0, false},
    {-12.562096, 62,  0, false},
    {-13.040989, 59,  0, false},
    {-13.484814, 58,  0, false},
    {-14.048302, 53,  0, false},
    {-14.598064, 52,  0, false},
    {-15.188733, 49,  0, false},
    {-15.656734, 48,  3, true},
    {-16.176632, 45,  0, false},
    {-16.55401, 44,  0, false},
    {-17.186821, 55,  0, false},
    {-17.73037, 54,  0, false},
    {-18.323431, 51,  0, false},
    {-18.797075, 50,  0, false},
    {-19.320236, 47,  0, false},
    {-19.736372, 46,  0, false},
    {-20.222572, 41,  0, false},
    {-20.787691, 40,  0, false},
    {-21.318125, 37,  0, false},
    {-21.935509, 36,  0, false},
    {-22.437605, 33,  0, false},
    {-22.856577, 32,  0, false},
    {-23.322384, 43, 0, false},
    {-23.971016, 42,  0, false},
    {-24.506605, 39,  0, false},
    {-24.999201, 38,  0, true},
};

int angle_mapping_cmp(const void *key, const void *item) {
    float a = *reinterpret_cast<const float *>(key);
    float b = reinterpret_cast<const angle_mapping *>(item)->angle;

    if(a < b) {
        return 1;
    } else if(a > b) {
        return -1;
    } else {
        return 0;
    }
}

bool point_allowed(point &p) {
    float angle = p.angle();

    size_t item = 0;
    float error = 9999999;
    for( ; item < 64; item++) {
        float cur_error = abs(angle - angle_mappings[item].angle);
        if(cur_error < error) {
            error = cur_error;
        } else {
            break;
        }
    }

    if(item) {
        item--;
    }

    const angle_mapping *mapping = &angle_mappings[item];

    if(error > angle_delta) {
        return false;
    }

    return mapping->keep;
}

std::vector<point> filter_points(vector<point> &input) {
    vector<point> output;
    float min_angle = 0;
    float max_angle = 0;

    for(auto &point: input) {
        min_angle = min(min_angle, point.angle());
        max_angle = max(max_angle, point.angle());
        if(point_allowed(point)) {
            output.push_back(point);
        }
    }

    if(verbose) {
        cout << "Kept " << (100.0 * output.size() / (float) input.size()) << endl;
    }

    return output;
}

void elevate_points(vector<point> &input) {
    for(auto &point: input) {
        point.z = point.z - 1.55;
    }
}

void filter_point_file(const fs::path &path, const directory_pair &pair) {
    // read source points in completely
    vector<point> source_data;
    ifstream source_file(path, ios::binary);
    point data;
    while(source_file.read(reinterpret_cast<char *>(&data), sizeof(data))) {
        source_data.push_back(data);
    }

    // filter points
    vector<point> target_data = filter_points(source_data);

    // elevate points by 1.55 (from the perspective of the Lidar: So we need to substract the height from the point's z-value!)
    elevate_points(target_data);

    // write points out
    fs::path target_path = pair.target / path.filename();
    ofstream target_file(target_path, ios::binary);
    for(auto &p: target_data) {
        if(!target_file.write(reinterpret_cast<char *>(&p), sizeof(p))) {
            cerr << "Error writing out file!" << endl;
            exit(1);
        }
    }
}

// label files are TXT files with one label per line. every line
// contains a label (as text), followed by parameters. this struct
// represents such a label line.
//    1    type         Describes the type of object: 'Car', 'Van', 'Truck',
//                      'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram',
//                      'Misc' or 'DontCare'
//    1    truncated    Float from 0 (non-truncated) to 1 (truncated), where
//                      truncated refers to the object leaving image boundaries
//    1    occluded     Integer (0,1,2,3) indicating occlusion state:
//                      0 = fully visible, 1 = partly occluded
//                      2 = largely occluded, 3 = unknown
//   1    alpha        Observation angle of object, ranging [-pi..pi]
//    4    bbox         2D bounding box of object in the image (0-based index):
//                      contains left, top, right, bottom pixel coordinates
//    3    dimensions   3D object dimensions: height, width, length (in meters)
//    3    location     3D object location x,y,z in camera coordinates (in meters)
//    1    rotation_y   Rotation ry around Y-axis in camera coordinates [-pi..pi]
//    1    score        Only for results: Float, indicating confidence in
//                      detection, needed for p/r curves, higher is better.
struct label_item {
    std::string type;
    float truncated;
    float occluded;
    float alpha;
    float bbox[4];
    float dimensions[3];
    float location[3];
    float rotation_y;
    // float score;

    friend ostream& operator<<(ostream& o, const label_item& i) {
        o << i.type << " " << i.truncated << " " << i.occluded << " " << i.alpha << " ";
        o << i.bbox[0] << " " << i.bbox[1] << " " << i.bbox[2] << " " << i.bbox[3] << " ";
        o << i.dimensions[0] << " " << i.dimensions[1] << " " << i.dimensions[2] << " ";
        o << i.location[0] << " " << i.location[1] << " " << i.location[2] << " ";
        o << i.rotation_y; // " " << i.score;
    }
};

bool keep_label(const label_item &item) {
    return true;
}

void elevate_label(label_item &item) {
    item.location[2] =  item.location[2] - 1.55;
}

void filter_label_file(const fs::path &path, const directory_pair &pair) {
    ifstream file(path);
    std::string line;
    vector<label_item> items;

    // where to write out label files
    fs::path target_path = pair.target / path.filename();
    ofstream target_file(target_path);

    // go through labels, parse label line and filter those out that are to be kept.
    while(std::getline(file, line)) {
        label_item item;
        std::stringstream linestream(line);
        linestream >> item.type;
        linestream >> item.truncated;
        linestream >> item.occluded;
        linestream >> item.alpha;
        linestream >> item.bbox[0] >> item.bbox[1] >> item.bbox[2] >> item.bbox[3];
        linestream >> item.dimensions[0] >> item.dimensions[1] >> item.dimensions[2];
        linestream >> item.location[0] >> item.location[1] >> item.location[2];
        linestream >> item.rotation_y;

        // make sure parsing line worked
        if(!linestream) {
            cout << "ERROR parsing line " << line << endl;
            exit(1);
        }

        elevate_label(item);
        target_file << item << endl;

    }
}

int test() {
    point a{0, 0, 1, 4};
    assert(a.angle() == 90);

    point b{0, 1, 1, 4};
    assert(b.angle() == 45);

    return 0;
}

void filter_points(const vector<directory_pair> &dirs) {
    for(auto &pair: dirs) {
        fs::create_directory(pair.target);

        // get list of paths
        deque<fs::path> source_files;
        for(auto &p: fs::directory_iterator(pair.source)) {
            source_files.push_back(p.path());
        }

        mutex source_files_lock;

        auto work = [&source_files_lock, &source_files, &pair] {
            while(true) {
                source_files_lock.lock();
                if(!source_files.size()) {
                    source_files_lock.unlock();
                    return;
                }

                auto path = source_files.front();
                source_files.pop_front();
                source_files_lock.unlock();

                filter_point_file(path, pair);
            }
        };

        // use as many threads as there are CPU cores in the system
        size_t thread_count = thread::hardware_concurrency();

        // launch thread pool with a shared work queue
        vector<thread> pool;
        for(size_t i = 0; i < thread_count; i++) {
            pool.push_back(thread(work));
        }

        // wait for all work to be done
        for(auto &thread: pool) {
            thread.join();
        }
    }
}

void filter_labels(const vector<directory_pair> &dirs) {
    for(auto &pair: dirs) {
        fs::create_directory(pair.target);

        // get list of paths
        deque<fs::path> source_files;
        for(auto &p: fs::directory_iterator(pair.source)) {
            source_files.push_back(p.path());
        }

        mutex source_files_lock;

        auto work = [&source_files_lock, &source_files, &pair] {
            while(true) {
                source_files_lock.lock();
                if(!source_files.size()) {
                    source_files_lock.unlock();
                    return;
                }

                auto path = source_files.front();
                source_files.pop_front();
                source_files_lock.unlock();

                filter_label_file(path, pair);
            }
        };

        // use as many threads as there are CPU cores in the system
        size_t thread_count = thread::hardware_concurrency();

        // launch thread pool with a shared work queue
        vector<thread> pool;
        for(size_t i = 0; i < thread_count; i++) {
            pool.push_back(thread(work));
        }

        // wait for all work to be done
        for(auto &thread: pool) {
            thread.join();
        }
    }
}

int main(int argc, char *argv[]) {
    unused(argc);
    unused(argv);

    test();

    filter_points(point_dirs);
    filter_labels(label_dirs);
}