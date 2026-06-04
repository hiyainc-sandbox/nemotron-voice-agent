// Thin executable wrapper for the session composition gate.
//
// The implementation now lives in lib/session/session.cpp so density_main and
// the production server can share a real runtime library boundary.
#include "lib/session/session.h"

int main(int argc, char** argv) {
  return session_main_entrypoint(argc, argv);
}
