# Executable to be built.
add_executable(addmul addmul.cpp)

# Link libraries to the target executable at build time
target_link_libraries(addmul util mymath)
