#include "stdio.h"
#include "Detector.hpp"
#include <iostream>
#include <stdexcept>

int Orbits::get_window(double t)
{
    int out = int(t / dt);
    if ((out < 0) || (out >= N))
        return -1;
    else
        return out;
}

int Orbits::get_link_ind(int link)
{
    if (link == 12)
        return 0;
    else if (link == 23)
        return 1;
    else if (link == 31)
        return 2;
    else if (link == 13)
        return 3;
    else if (link == 32)
        return 4;
    else if (link == 21)
        return 5;
    else
#ifdef __CUDACC__
        printf("BAD link ind. Must be 12, 23, 31, 13, 32, 21.");
#else
        throw std::invalid_argument("Bad link ind. Must be 12, 23, 31, 13, 32, 21.");
#endif // __CUDACC__
    return -1;
}

int Orbits::get_sc_ind(int sc)
{
    if (sc == 1)
        return 0;
    else if (sc == 2)
        return 1;
    else if (sc == 3)
        return 2;
    else
#ifdef __CUDACC__
        printf("BAD sc ind. Must be 1,2,3.");
#else
        throw std::invalid_argument("Bad sc ind. Must be 1,2,3.");
#endif // __CUDACC__
    return -1;
}

double Orbits::interpolate(double t, double *in_arr, int down_ind, int up_ind, int ndim, int pos)
{
    double down = in_arr[down_ind * ndim + pos];
    double up = in_arr[up_ind * ndim + pos];
    // m *(x - x0) + y0
    return ((up - down) / dt) * (t - (dt * down_ind)) + down;
}

void Orbits::get_normal_unit_vec_ptr(Vec *vec, double t, int link)
{
    Vec _tmp = get_normal_unit_vec(t, link);
    vec->x = _tmp.x;
    vec->y = _tmp.y;
    vec->z = _tmp.z;
}

Vec Orbits::get_normal_unit_vec(double t, int link)
{
    int window = get_window(t);
    if (window == -1)
    {
        // out of bounds
        return Vec(0.0, 0.0, 0.0);
    }

    int link_ind = get_link_ind(link);

    int up_ind = (window + 1) * nlinks + link_ind;
    int down_ind = window * nlinks + link_ind;

    // x (pos = 0) ndim = 3
    double x_out = interpolate(t, n_arr, down_ind, up_ind, 3, 0);
    // y (pos = 1)
    double y_out = interpolate(t, n_arr, down_ind, up_ind, 3, 1);
    // z (pos = 2)
    double z_out = interpolate(t, n_arr, down_ind, up_ind, 3, 2);

    return Vec(x_out, y_out, z_out);
}

double Orbits::get_light_travel_time(double t, int link)
{
    int window = get_window(t);
    if (window == -1)
    {
        // out of bounds
        return 0.0;
    }

    int link_ind = get_link_ind(link);

    int up_ind = (window + 1) * (nlinks + link_ind);
    int down_ind = window * (nlinks + link_ind);

    // x (pos = 0), ndim = 1
    double ltt_out = interpolate(t, ltt_arr, down_ind, up_ind, 1, 0);

    return ltt_out;
}

Vec Orbits::get_pos(double t, int sc)
{
    int window = get_window(t);
    if (window == -1)
    {
        // out of bounds
        return Vec(0.0, 0.0, 0.0);
    }

    int sc_ind = get_sc_ind(sc);

    int up_ind = (window + 1) * nspacecraft + sc_ind;
    int down_ind = window * nspacecraft + sc_ind;

    // x (pos = 0), ndim = 3
    double x_out = interpolate(t, x_arr, down_ind, up_ind, 3, 0);
    // y (pos = 1), ndim = 3
    double y_out = interpolate(t, x_arr, down_ind, up_ind, 3, 1);
    // z (pos = 2), ndim = 3
    double z_out = interpolate(t, x_arr, down_ind, up_ind, 3, 2);
    return Vec(x_out, y_out, z_out);
}

void Orbits::get_pos_ptr(Vec *vec, double t, int sc)
{
    Vec _tmp = get_pos(t, sc);
    vec->x = _tmp.x;
    vec->y = _tmp.y;
    vec->z = _tmp.z;
}