#pragma once

#include "tfStringBase.hpp"

namespace tf
{
	class StringW;

	class String : public StringBase<char, String>
	{
	public:
		using StringBase<char, String>::StringBase;

		StringW fromUtf8() const;
		void    toLower();
	};
}

namespace std
{
	template<> struct hash<tf::String>
	{
		size_t operator()(const tf::String& s) const noexcept
		{
			return std::hash<std::string>{}(s);
		}
	};
}