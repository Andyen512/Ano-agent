#include <stdatomic.h>

#include "jitprofiling.h"

static atomic_uint next_method_id = 1;

unsigned int JITAPI iJIT_GetNewMethodID(void) {
    return atomic_fetch_add(&next_method_id, 1);
}

iJIT_IsProfilingActiveFlags JITAPI iJIT_IsProfilingActive(void) {
    return iJIT_NOTHING_RUNNING;
}

int JITAPI iJIT_NotifyEvent(iJIT_JVM_EVENT event_type, void *event_data) {
    (void)event_type;
    (void)event_data;
    return 0;
}
