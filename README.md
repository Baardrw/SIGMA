 real_t t = i * 2.0f; // Parameter along line
    Eigen::Vector3f hit(true_x0 + t * dx, true_y0 + t * dy, t * dz);

    h_data.sensor_pos_x[i] = hit.x();
    h_data.sensor_pos_y[i] = hit.y();
    h_data.sensor_pos_z[i] = hit.z();