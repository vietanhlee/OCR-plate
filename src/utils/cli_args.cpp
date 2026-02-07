#include "utils/cli_args.h"

#include <iostream>
#include <string>

#include "app_config.h"

namespace cli_args {

void PrintUsage(const char* argv0, std::ostream& os) {
	os
		<< "Cach dung:\n"
		<< "  " << argv0 << " --image <duong_dan_anh.jpg>\n\n"
		<< "Ghi chu:\n"
		<< "  - Model duoc fix cung: " << app_config::kModelPath << "\n"
		<< "  - Tien xu ly: doc anh -> RGB -> resize (" << app_config::kInputW << "x" << app_config::kInputH << ") -> uint8 NHWC\n"
		<< "  - Hau xu ly: map index -> ky tu va xoa ky tu '_' o cuoi (blank/pad)\n";
}

Options Parse(int argc, char** argv) {
	Options opt;
	for (int i = 1; i < argc; ++i) {
		std::string a = argv[i];
		if ((a == "--image" || a == "-i") && i + 1 < argc) {
			opt.image_path = argv[++i];
		} else if (a == "--help" || a == "-h") {
			opt.show_help = true;
			return opt;
		} else {
			throw std::runtime_error("Tham so khong hop le: " + a);
		}
	}
	return opt;
}

} // namespace cli_args
