FA-QEM for TripoSG
------------------
faqem/              FA-QEM code (do not edit)
faqem_adapter.py    the one function to call: simplify_mesh_faqem(vertices, faces, detail="Medium", target_faces=None)
requirements_faqem.txt
smoke_test.py       python smoke_test.py [mesh.glb]  -> must print "SMOKE TEST PASSED"
See FAQEM_TripoSG_integration.md (sent alongside this zip) for the integration steps.

C++ engine (optional, same results)
-----------------------------------
faqem_cpp*.so  the compiled C++ engine (build it in the FA-QEM project: see cpp/README.md, then copy
               cpp/build/faqem_cpp.cpython-312-darwin.so here). When it can be imported,
               simplify_mesh_faqem() uses it; results are bit-identical to faqem/ (same face counts,
               same vertices, same deviation), and info["engine"] says which engine ran.
               FAQEM_BACKEND=python forces the Python engine, FAQEM_BACKEND=cpp requires C++.
               The .so is built for macOS arm64 / Python 3.12. Linux is not supported yet (the engine
               replays numpy's Apple-Accelerate calls; Linux needs an OpenBLAS port). Without the
               .so the adapter simply uses the Python engine.
