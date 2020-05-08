//        $Id: pfc_bitmap_3.h 40278 2020-02-28 18:46:06Z p20068 $
//       $URL: https://svn01.fh-hagenberg.at/bin/cepheiden/vocational/teaching/SE-Master/MPV3/2018-WS/ILV/src/bitmap/src/pfc_bitmap_3.h $
//  $Revision: 40278 $
//      $Date: 2020-02-28 19:46:06 +0100 (Fr., 28 Feb 2020) $
//    $Author: p20068 $
//
//    Creator: Peter Kulczycki (peter.kulczycki<AT>fh-hagenberg.at)
//   Creation:
//  Copyright: (c) 2020 Peter Kulczycki (peter.kulczycki<AT>fh-hagenberg.at)
//
//    License: This document contains proprietary information belonging to
//             University of Applied Sciences Upper Austria, Campus
//             Hagenberg. It is distributed under the Boost Software License,
//             Version 1.0 (see http://www.boost.org/LICENSE_1_0.txt).
//
// Annotation: This file is part of the code snippets handed out during one
//             of my HPC lessons held at the University of Applied Sciences
//             Upper Austria, Campus Hagenberg.

#pragma once

#include "./pfc_base.h"

#undef  PFC_BITMAP_VERSION
#define PFC_BITMAP_VERSION "3.2.0"

#if defined PFC_DETECTED_COMPILER_NVCC
   #error PFC: 'pfc_bitmap_3.h' cannot be compiled with nvcc (maybe use 'pfc_types.h' instead)
#endif

#if !defined PFC_HAVE_GSL
   #error PFC: 'Guideline Support Library' is required
#endif

namespace pfc {

// -------------------------------------------------------------------------------------------------

class bitmap_exception final : public std::runtime_error {
   using inherited = std::runtime_error;

   public:
      explicit bitmap_exception (std::string msg) noexcept : inherited {std::move (msg)} {
      }
};

// -------------------------------------------------------------------------------------------------

class bitmap final {
   #pragma pack (push, 1)

   struct file_header_t {
      word_t  type;         // file type; must be 0x4d42 (i.e. 'BM')
      dword_t size;         // size, in bytes, of the bitmap file
      word_t  reserved_1;   // reserved; must be 0
      word_t  reserved_2;   // reserved; must be 0
      dword_t offset;       // offset, in bytes, from the beginning of the 'file_header_t' to the bitmap bits
   };

   struct info_header_t {
      dword_t size;            // number of bytes required by the structure
      long_t  width;           // width of the bitmap, in pixels
      long_t  height;          // height of the bitmap, in pixels
      word_t  planes;          // number of planes for the target device; must be 1
      word_t  bit_count;       // number of bits per pixel
      dword_t compression;     // type of compression; 0 for uncompressed RGB
      dword_t size_image;      // size, in bytes, of the pixel_span
      long_t  x_pels_pm;       // horizontal resolution, in pixels per meter
      long_t  y_pels_pm;       // vertical resolution, in pixels per meter
      dword_t clr_used;        // number of color indices in the color table
      dword_t clr_important;   // number of color indices that are considered important
   };

   #pragma pack (pop)

// using pixel_t           = BGR_4_t;
// using pixel_file_t      = BGR_3_t;
   using pixel_span_t      = gsl::span <pixel_t>;
   using pixel_file_span_t = gsl::span <pixel_file_t>;

   static constexpr int c_size_file_header {sizeof (file_header_t)};
   static constexpr int c_size_info_header {sizeof (info_header_t)};
   static constexpr int c_size_pixel       {sizeof (pixel_t)};
   static constexpr int c_size_pixel_file  {sizeof (pixel_file_t)};

   static_assert (c_size_file_header == 14);
   static_assert (c_size_info_header == 40);
   static_assert (c_size_pixel       ==  4);
   static_assert (c_size_pixel_file  ==  3);

   public:
      using pixel_type      = pixel_t;
      using pixel_span_type = pixel_span_t;

      bitmap () {
         create (0, 0);
      }

      explicit bitmap (int const width, int const height, pixel_span_t pixel_span = {}, bool const clear_span = true) {
         create (width, height, std::move (pixel_span), clear_span);
      }

      explicit bitmap (std::string const & name) {
         from_file (name);
      }

      bitmap (bitmap const & src) {
         create (src.m_width, src.m_height); memcpy (m_pixel_span, src.m_pixel_span);
      }

      bitmap (bitmap && tmp) noexcept
         : m_file_header {tmp.m_file_header}
         , m_info_header {tmp.m_info_header}
         , m_height      {tmp.m_height}
         , m_width       {tmp.m_width}
         , m_p_image     {std::move (tmp.m_p_image)}
         , m_pixel_span  {std::move (tmp.m_pixel_span)} {
      }

     ~bitmap () = default;

      bitmap & operator = (bitmap const & rhs) {
         if (&rhs != this) {
            create (rhs.m_width, rhs.m_height); memcpy (m_pixel_span, rhs.m_pixel_span);
         }

         return *this;
      }

      bitmap & operator = (bitmap && tmp) noexcept {
         if (&tmp != this) {
            m_file_header = tmp.m_file_header;
            m_info_header = tmp.m_info_header;
            m_height      = tmp.m_height;
            m_width       = tmp.m_width;

            std::swap (m_p_image,    tmp.m_p_image);
            std::swap (m_pixel_span, tmp.m_pixel_span);
         }

         return *this;
      }

      auto const & height () const noexcept {
         return m_height;
      }

      auto & pixel_span () noexcept {
         return m_pixel_span;
      }

      auto const & pixel_span () const noexcept {
         return m_pixel_span;
      }

      auto size () const noexcept {
         return m_width * m_height;
      }

      auto const & width () const noexcept {
         return m_width;
      }

      auto & at (int const x, int const y) noexcept {
         return m_pixel_span[y * m_width + x];
      }

      auto const & at (int const x, int const y) const noexcept {
         return m_pixel_span[y * m_width + x];
      }

      auto * data () noexcept {
         return std::data (m_pixel_span);
      }

      auto const * data () const noexcept {
         return std::data (m_pixel_span);
      }

      void clear () {
         create (0, 0);
      }

      void create (int const width, int const height, pixel_span_t pixel_span = {}, bool const clear_span = true) {
         m_width  = std::max (0, adjust_width (width));
         m_height = std::max (0, height);

         memset (m_file_header, 0);
         memset (m_info_header, 0);

         m_file_header.type   = 0x4d42;
         m_file_header.size   = c_size_file_header + c_size_info_header + size () * c_size_pixel_file;
         m_file_header.offset = c_size_file_header + c_size_info_header;

         m_info_header.size       = c_size_info_header;
         m_info_header.width      = m_width;
         m_info_header.height     = m_height;
         m_info_header.planes     =  1;
         m_info_header.bit_count  = 24;
         m_info_header.size_image = size () * c_size_pixel_file;

         if (!is_valid (m_file_header) || !is_valid (m_info_header)) {
            throw bitmap_exception {"Invalid bitmap header(s)."};
         }

         if (pixel_span.empty ()) {
            m_p_image    = std::make_unique <pixel_t []> (size ());
            m_pixel_span = {m_p_image.get (), static_cast <std::size_t> (size ())};
         } else {
            m_p_image    = nullptr;
            m_pixel_span = std::move (pixel_span);
         }

         if (std::size (m_pixel_span) < size ()) {
            throw bitmap_exception {
               "Pixel span too small (need space for " + std::to_string (size ()) + " pels, got " + std::to_string (std::size (m_pixel_span)) + ")."
            };
         }

         if (clear_span) {
            memset (m_pixel_span, 0xff);
         }
      }

      void from_file (std::string const & name) {
         std::ifstream in {name, std::ios_base::binary};
         auto          ok {true};

         clear ();

         file_header_t file_header {};
         info_header_t info_header {};

         ok = ok && in;
         ok = ok && read (in, file_header);
         ok = ok && read (in, info_header);
         ok = ok && is_valid (file_header);
         ok = ok && is_valid (info_header);

         if (ok) {
            create (info_header.width, info_header.height); ok = (size () == pixel_span_from_stream (in, m_pixel_span, m_width));
         }

         if (!ok) {
            throw bitmap_exception {"Error reading bitmap from file '" + name + "'."};
         }
      }

      void to_file (std::string const & name) const {
         std::ofstream out {name, std::ios_base::binary};

         if (!out ||
             !write (out, m_file_header) ||
             !write (out, m_info_header) ||
             !(size () == pixel_span_to_stream (out, m_pixel_span, m_width))) {
            throw bitmap_exception {"Error writing bitmap to file '" + name + "'."};
         }
      }

   private:
      /**
       * The threefold width of the bitmap must be evenly dividable by four.
       */
      static constexpr int adjust_width (int const width) noexcept {
         return std::max (0, ceil_div (width * c_size_pixel_file, 4) * 4 / c_size_pixel_file);
      }

      static constexpr bool is_valid (file_header_t const & hdr) noexcept {
         return  (hdr.offset     == c_size_file_header + c_size_info_header) &&
                 (hdr.reserved_1 == 0) &&
                 (hdr.reserved_2 == 0) &&
                !(hdr.size       <  c_size_file_header + c_size_info_header) &&
                 (hdr.type       == 0x4d42);
      }

      static bool is_valid (info_header_t const & hdr) {
         return (hdr.bit_count     == 24) &&
                (hdr.clr_important ==  0) &&
                (hdr.clr_used      ==  0) &&
                (hdr.compression   ==  0) &&
                (hdr.planes        ==  1) &&
                (hdr.size          == c_size_info_header) &&
                (hdr.size_image    == gsl::narrow <dword_t> (hdr.width * hdr.height * c_size_pixel_file)) &&
                (hdr.width         == adjust_width (hdr.width));
      }

      static int pixel_span_from_stream (std::istream & in, pixel_span_t const & pixel_span, int const chunk_size) noexcept {
//       std::vector <pixel_file_t> buffer      (chunk_size);
         auto                       buffer      {std::make_unique <pixel_file_t []> (chunk_size)};
         auto                       buffer_span {pixel_file_span_t {buffer.get (), static_cast <std::size_t> (chunk_size)}};
         auto                       i           {0};
         auto                       processed   {0};

         for (auto & pel : pixel_span) {
            i %= chunk_size;

            if ((i == 0) && read (in, buffer_span)) {
               processed += chunk_size;
            }

            pel.bgr_3 = buffer_span[i++];
         }

         return processed;
      }

      static int pixel_span_to_stream (std::ostream & out, pixel_span_t const & pixel_span, int const chunk_size) noexcept {
//       std::vector <pixel_file_t> buffer      (chunk_size);
         auto                       buffer      {std::make_unique <pixel_file_t []> (chunk_size)};
         auto                       buffer_span {pixel_file_span_t {buffer.get (), static_cast <std::size_t> (chunk_size)}};
         auto                       i           {0};
         auto                       processed   {0};

         for (auto const & pel : pixel_span) {
            buffer_span[i++] = pel.bgr_3;

            if ((i == chunk_size) && write (out, buffer_span)) {
               processed += chunk_size;
            }

            i %= chunk_size;
         }

         return processed;
      }

      file_header_t                m_file_header {};
      info_header_t                m_info_header {};
      int                          m_height      {0};
      int                          m_width       {0};
      std::unique_ptr <pixel_t []> m_p_image     {};
      pixel_span_t                 m_pixel_span  {};
};

// -------------------------------------------------------------------------------------------------

class bitmap_group final {
   public:
      explicit bitmap_group (std::size_t const n, int const width, int const height) : m_group (n) {
         for (auto & bmp : m_group) {
            bmp.create (width, height);
         }
      }

      template <typename G> explicit bitmap_group (G const & memory_group, int const width, int const height) {
         m_group.reserve (std::size (memory_group));

         for (auto const & m : memory_group) {
            m_group.emplace_back (width, height, bitmap::pixel_span_type {m.get (), static_cast <std::size_t> (width * height)});
         }
      }

      bitmap_group (bitmap_group const &) = delete;
      bitmap_group (bitmap_group &&) = default;

      bitmap_group & operator = (bitmap_group const &) = delete;
      bitmap_group & operator = (bitmap_group &&) = default;

      auto & operator [] (std::size_t const i) {
         return m_group[i];
      }

      auto & at (std::size_t const i) {
         return m_group.at (i);
      }

      auto * data (std::size_t const i) {
         return at (i).data ();
      }

      void to_file (std::string const & prefix, std::size_t i = 0) {
         for (auto const & bmp : m_group) {
            bmp.to_file (filename (prefix, i++));
         }
      }

   private:
      std::string filename (std::string name, std::size_t const i) const {
//       static auto const n {digits (std::size (m_group))};

         name += '-';
//       name += std::string (n - digits (i), '0');
         name += std::to_string (i);
         name += ".bmp";

         return name;
      }

      std::vector <bitmap> m_group;
};

// -------------------------------------------------------------------------------------------------

}   // namespace pfc
