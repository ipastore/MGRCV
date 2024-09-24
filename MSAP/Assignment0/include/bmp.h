#pragma once
#include <vector>
#include <iostream>
#include <fstream>

class RGB {
    unsigned char b_, g_, r_;
public:
    RGB(unsigned char r, unsigned char g, unsigned char b) : b_(b), g_(g), r_(r)  {}
    RGB(unsigned char v = 0) : b_(v), g_(v), r_(v) {}
    unsigned char r() const { return r_; }
    unsigned char g() const { return g_; }
    unsigned char b() const { return b_; }
};

class Image {
public:
    constexpr static int header_size = 14;
    constexpr static int info_header_size = 40;
    constexpr static size_t bytes_per_pixel = 3;

    Image(size_t width, size_t height) :
        image_px_width{ width }, image_px_height{ height }, row_width{ image_px_width * bytes_per_pixel },
        row_padding{ (4 - row_width % 4) % 4 }, row_stride{ row_width + row_padding }, file_size{ header_size + info_header_size + (image_px_height * row_stride) },
        image(image_px_height, std::vector<RGB>(image_px_width))
    {
        //header file type
        file_header[0] = 'B';
        file_header[1] = 'M';


        //header file size info
        file_header[2] = static_cast<unsigned char>(file_size);
        file_header[3] = static_cast<unsigned char>(file_size >> 8);
        file_header[4] = static_cast<unsigned char>(file_size >> 16);
        file_header[5] = static_cast<unsigned char>(file_size >> 24);

        //header offset to pixel data
        file_header[10] = header_size + info_header_size;

        //info header size
        info_header[0] = info_header_size;

        //info header image width
        info_header[4] = static_cast<unsigned char>(image_px_width);
        info_header[5] = static_cast<unsigned char>(image_px_width >> 8);
        info_header[6] = static_cast<unsigned char>(image_px_width >> 16);
        info_header[7] = static_cast<unsigned char>(image_px_width >> 24);

        //info header image height
        info_header[8] = static_cast<unsigned char>(image_px_height);
        info_header[9] = static_cast<unsigned char>(image_px_height >> 8);
        info_header[10] = static_cast<unsigned char>(image_px_height >> 16);
        info_header[11] = static_cast<unsigned char>(image_px_height >> 24);

        //info header planes
        info_header[12] = 1;

        //info header bits per pixel
        info_header[14] = 8 * bytes_per_pixel;
    }

    size_t width() const {
        return image_px_width;
    }

    size_t height() const {
        return image_px_height;
    }

    const RGB& operator()(std::size_t x, std::size_t y) const {
        return image[y][x];
    }
    RGB& operator()(std::size_t x, std::size_t y) {
        return image[y][x];
    }

    void save(const char* file_name) const {
        std::ofstream img_file(file_name, std::ios_base::binary | std::ios_base::out);

        img_file.write((char*)file_header, header_size);
        img_file.write((char*)info_header, info_header_size);

        std::vector<char> allignment(row_padding);

        for (int y = image_px_height - 1; y >= 0; --y) {
            img_file.write((char*)image[y].data(), row_width);

            img_file.write(allignment.data(), row_padding);
        }

        img_file.close();
    }
private:
    size_t image_px_width;
    size_t image_px_height;

    size_t row_width;

    size_t row_padding;

    size_t row_stride;

    size_t file_size;

    unsigned char file_header[header_size] = { 0 };
    unsigned char info_header[info_header_size] = { 0 };
    std::vector<std::vector<RGB>> image;
};