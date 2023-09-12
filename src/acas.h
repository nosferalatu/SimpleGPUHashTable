/* Modifications Copyright (C) 2023 Intel Corporation
 *
 * This Program is subject to the terms of The Unlicense.​
 * If a copy of the license was not distributed with this file, ​
 * you can obtain one at https://spdx.org/licenses/Unlicense.html​
 *​
 *
 * SPDX-License-Identifier: Unlicense
 */

//===----------------------------------------------------------------------===//
//
// Following code is copied from atomic.hpp of dpct
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/sycl.hpp>

namespace acas {

/// Atomically compare the value at \p addr to the value expected and exchange
/// with the value desired if the value at \p addr is equal to the value expected.
/// Returns the value at the \p addr before the call.
/// \param [in, out] addr Multi_ptr.
/// \param expected The value to compare against the value at \p addr.
/// \param desired The value to assign to \p addr if the value at \p addr is expected.
/// \param success The memory ordering used when comparison succeeds.
/// \param fail The memory ordering used when comparison fails.
/// \returns The value at the \p addr before the call.
template <typename T, sycl::access::address_space addressSpace = sycl::access::address_space::global_space>
T atomic_compare_exchange_strong(
    sycl::multi_ptr<T, sycl::access::address_space::global_space> addr,
    T expected,
    T desired,
    sycl::memory_order success = sycl::memory_order::relaxed,
    sycl::memory_order fail    = sycl::memory_order::relaxed
) {
    // sycl::atomic_ref<T, addressSpace> obj(addr);
    sycl::atomic_ref<T, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space> obj(addr[0]);
    obj.compare_exchange_strong(expected, desired, success, fail);
    return expected;
}

/// Atomically compare the value at \p addr to the value expected and exchange
/// with the value desired if the value at \p addr is equal to the value expected.
/// Returns the value at the \p addr before the call.
/// \param [in] addr The pointer to the data.
/// \param expected The value to compare against the value at \p addr.
/// \param desired The value to assign to \p addr if the value at \p addr is expected.
/// \param success The memory ordering used when comparison succeeds.
/// \param fail The memory ordering used when comparison fails.
/// \returns The value at the \p addr before the call.
template <typename T, sycl::access::address_space addressSpace = sycl::access::address_space::global_space>
T atomic_compare_exchange_strong(
    T* addr,
    T expected,
    T desired,
    sycl::memory_order success = sycl::memory_order::relaxed,
    sycl::memory_order fail    = sycl::memory_order::relaxed
) {
    return atomic_compare_exchange_strong(sycl::multi_ptr<T, addressSpace>(addr), expected, desired, success, fail);
}

} // namespace acas
