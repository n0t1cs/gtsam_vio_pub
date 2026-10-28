// gvio-publish-check: run PublishFrame (the conversion the MPA bridge applies before vio_data_t) over a
// run's frames.csv and write the published trajectory, so the frame convention can be checked against
// ground truth without MPA:  gvio-publish-check LOG_DIR > published.csv
// Columns: t_ns,session,state,px,py,pz,qw,qx,qy,qz,vx,vy,vz,cov_valid,cov_xx,cov_yy,cov_zz
#include "publish_frame.h"

#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 2) { std::cerr << "usage: gvio-publish-check LOG_DIR\n"; return 2; }
    std::ifstream f(std::string(argv[1]) + "/frames.csv");
    if (!f) { std::cerr << "no frames.csv in " << argv[1] << "\n"; return 1; }
    std::string line;
    std::getline(f, line);
    gvio::PublishFrame pf(true);
    std::printf("t_ns,session,state,px,py,pz,qw,qx,qy,qz,vx,vy,vz,cov_valid,cov_xx,cov_yy,cov_zz\n");
    while (std::getline(f, line)) {
        std::vector<double> v;
        std::stringstream ss(line);
        std::string x;
        while (std::getline(ss, x, ',')) v.push_back(std::stod(x));
        if (v.size() < 23) continue;
        gvio::EngineOutput o;
        o.t_ns = (int64_t)v[0];
        o.session = (int)v[2];
        o.state = (int)v[3];
        o.is_keyframe = v[5] != 0;
        const Eigen::Quaterniond q(v[10], v[11], v[12], v[13]);
        o.T_wb.block<3, 3>(0, 0) = q.normalized().toRotationMatrix();
        o.T_wb.block<3, 1>(0, 3) = Eigen::Vector3d(v[7], v[8], v[9]);
        o.v_w = Eigen::Vector3d(v[14], v[15], v[16]);
        const gvio::PublishedState s = pf.convert(o);
        if (!s.valid) continue;
        const Eigen::Quaterniond qp(s.R);
        std::printf("%lld,%d,%d,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.6f,%.6f,%.6f,%d,%g,%g,%g\n",
                    (long long)o.t_ns, o.session, o.state, s.p.x(), s.p.y(), s.p.z(), qp.w(), qp.x(), qp.y(), qp.z(),
                    s.v.x(), s.v.y(), s.v.z(), s.cov_valid ? 1 : 0,
                    s.cov_valid ? s.pose_cov21[0] : 0.0, s.cov_valid ? s.pose_cov21[6] : 0.0,
                    s.cov_valid ? s.pose_cov21[11] : 0.0);
    }
    return 0;
}
