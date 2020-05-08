//        $Id: pfc_types.h 40278 2020-02-28 18:46:06Z p20068 $
//       $URL: https://svn01.fh-hagenberg.at/bin/cepheiden/vocational/teaching/SE-Master/MPV3/2018-WS/ILV/src/bitmap/src/pfc_types.h $
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

#include "./pfc_macros.h"

#include <cstddef>
#include <cstdint>

// -------------------------------------------------------------------------------------------------

namespace pfc {

using byte_t  = std::uint8_t;    // std::byte
using dword_t = std::uint32_t;   //
using long_t  = std::int32_t;    //
using word_t  = std::uint16_t;   //

}   // namespace pfc

PFC_STATIC_ASSERT (sizeof (pfc::byte_t)  == 1);
PFC_STATIC_ASSERT (sizeof (pfc::dword_t) == 4);
PFC_STATIC_ASSERT (sizeof (pfc::long_t)  == 4);
PFC_STATIC_ASSERT (sizeof (pfc::word_t)  == 2);

// -------------------------------------------------------------------------------------------------

namespace pfc {

#pragma pack (push, 1)
   struct BGR_3_t final {
      byte_t blue;
      byte_t green;
      byte_t red;
   };
#pragma pack (pop)

#if defined PFC_KNOW_PRAGMA_WARNING_PUSH_POP
#pragma warning (push)
#pragma warning (disable: 4201)   // nameless struct/union
#endif

#pragma pack (push, 1)
   struct BGR_4_t final {
      union {
         BGR_3_t bgr_3;

         struct {
            byte_t blue;
            byte_t green;
            byte_t red;
         };
      };

      byte_t unused;
   };
#pragma pack (pop)

#if defined PFC_KNOW_PRAGMA_WARNING_PUSH_POP
#pragma warning (pop)
#endif

using pixel_t      = BGR_4_t;
using pixel_file_t = BGR_3_t;

}   // namespace pfc

PFC_STATIC_ASSERT (sizeof (pfc::BGR_3_t) == 3);
PFC_STATIC_ASSERT (sizeof (pfc::BGR_4_t) == 4);
