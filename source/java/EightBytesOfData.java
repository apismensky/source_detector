/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.tika.parser.microsoft.onenote.fsshttpb.property;

import java.util.Arrays;
import java.util.List;

import org.apache.tika.parser.microsoft.onenote.fsshttpb.util.ByteUtil;


/**
 * This class is used to represent the property contains 8 bytes of data in the PropertySet.rgData stream field.
 */
public class EightBytesOfData implements IProperty {
    public byte[] data;

    /**
     * This method is used to deserialize the EightBytesOfData from the specified byte array and start index.
     *
     * @param byteArray  Specify the byte array.
     * @param startIndex Specify the start index from the byte array.
     * @return Return the length in byte of the EightBytesOfData.
     */
    public int doDeserializeFromByteArray(byte[] byteArray, int startIndex) {
        this.data = Arrays.copyOfRange(byteArray, startIndex, startIndex + 8);
        return 8;
    }

    /**
     * This method is used to convert the element of EightBytesOfData into a byte List.
     *
     * @return Return the byte list which store the byte information of EightBytesOfData.
     */
    public List<Byte> serializeToByteList() {
        return ByteUtil.toListOfByte(this.data);
    }
}
