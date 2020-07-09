
#include <iostream>
#include <string>
#include <fstream>
#include <vector>

using namespace std;
static bool readConfigFile(const char * cfgfilepath, const string & key, vector<double>  & nums)
{
	fstream cfgFile;
	cfgFile.open(cfgfilepath);//���ļ�	
	if (!cfgFile.is_open())
	{
		cout << "can not open cfg file!" << endl;
		return false;
	}

	nums.clear();
	char tmp[1000];
	while (!cfgFile.eof())//ѭ����ȡÿһ��
	{
		cfgFile.getline(tmp, 1000);//ÿ�ж�ȡǰ1000���ַ���1000��Ӧ���㹻��
		string line(tmp);
		size_t pos = line.find('=');//�ҵ�ÿ�еġ�=����λ�ã�֮ǰ��key֮����value
		if (pos == string::npos) return false;
		string tmpKey = line.substr(0, pos);//ȡ=��֮ǰ
		string value;
		char s[30];
		
		double x;
		if (key == tmpKey)
		{
			//string x;
			////while (ss >> x)
			//cfgFile >> x;
			
			value = line.substr(pos + 1);//ȡ=��֮��
			int j = 0;
			for (int i = 0; i < value.length(); i++)
			{
				
				if (value[i] == ',' )
				{
					i++;
					j = 0;
					sscanf_s(s, "%lf", &x);  //%f��Ӧfloat, %lf��Ӧdouble.
					nums.push_back(x);
					memset(s, 0, sizeof(s));
				}
				s[j] = value[i];
				j++;
				if (i == value.length()-1)
				{
					sscanf_s(s, "%lf", &x);
					nums.push_back(x);
				}

			}
			return true;
		}
	}
	return false;
}
